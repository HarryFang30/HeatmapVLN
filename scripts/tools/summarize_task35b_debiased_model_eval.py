#!/usr/bin/env python3
"""Strict matched summary for the six Task-3.5b debiased model evaluations.

The input reports are evaluation-only runs of ``diagnose_heatmap_shortcuts``:
the shared Task-4 step-0 head, branch-B at steps 25 and 100, branch-C at
step 100, and the independently trained 500-step Full/No-input heads.  This
script treats exact sample pairing as part of the experimental contract.  It
does not silently summarize partial, reordered, retrained, or checkpoint-
aliased runs.

The visual-grounding gate is deliberately conservative.  Aggregate shortcut
suppression and slot-by-view support are reported as separate claims because
the Task-3.5b selection may have no non-back positives for the recent history
slot even when its aggregate marginal balance is improved.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import random
import re
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path, PurePosixPath
from typing import Any

REPORT_ROLES = (
    "step0",
    "b25",
    "b100",
    "c100",
    "head500-full",
    "head500-noinput",
)
FULL_ROLES = REPORT_ROLES[:-1]
EXPECTED_FULL_EVALUATIONS = (
    "standard",
    "zero-pose",
    "blank-images",
    "history-shuffle",
    "current-shuffle",
    "pose-conflict",
    "pose-conflict-shifted-target",
)
EXPECTED_NO_INPUT_EVALUATIONS = ("standard",)
SELECTION_ALGORITHM = "task35b_verified_explicit_manifest_v1"
SELECTION_SCHEMA = "task35b_debiased_selection_v1"
EXPECTED_VAL_SAMPLES = 64
EXPECTED_LORA_TENSORS = 224
VIEW_NAMES = ("front", "right", "back", "left")

DEFAULT_EMPIRICAL_PRIOR_JOINT_PCK8 = 0.271845
SIGNIFICANCE_ALPHA = 0.05
MIN_PRIOR_JOINT_PCK8_DELTA = 0.10
CLUSTER_BOOTSTRAP_REPLICATES = 5000
MIN_BLANK_JOINT_PCK8_DROP = 0.10
MIN_BLANK_PAIRED_CHANGE_FRACTION = 0.25
MIN_SHUFFLE_LOCALIZATION_DROP = 0.05
MIN_SHUFFLE_PAIRED_CHANGE_FRACTION = 0.25

SUMMARY_METRICS = (
    "loss",
    "visibility_auroc",
    "visibility_auprc",
    "visibility_f1",
    "visibility_precision",
    "visibility_recall",
    "visible_view_accuracy",
    "visible_history_count",
    "visible_view_count",
    "median_pixel_error",
    "median_u_error",
    "pck4",
    "pck8",
    "joint_median_pixel_error",
    "joint_pck4",
    "joint_pck8",
    "samples",
)
RECOMPUTED_METRICS = (
    "visible_view_accuracy",
    "visible_history_count",
    "visible_view_count",
    "median_pixel_error",
    "median_u_error",
    "pck4",
    "pck8",
    "joint_median_pixel_error",
    "joint_pck4",
    "joint_pck8",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    for role in REPORT_ROLES:
        parser.add_argument(f"--{role}-report", required=True)
    parser.add_argument(
        "--selection-report",
        default=None,
        help=(
            "Optional Task-3.5b selection report. Its exact debiased val-64 "
            "identity is checked, its after-debias empirical prior is used, "
            "and its recent-slot view support is reported."
        ),
    )
    parser.add_argument(
        "--empirical-prior-joint-pck8",
        type=float,
        default=None,
        help=(
            "Empirical-prior null when --selection-report is omitted. Defaults "
            f"to the recorded Task-3.5b value {DEFAULT_EMPIRICAL_PRIOR_JOINT_PCK8}."
        ),
    )
    parser.add_argument("--output-dir", required=True)
    return parser.parse_args()


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected JSON object: {path}")
    return payload


def _sha256_lines(values: Iterable[str]) -> str:
    return hashlib.sha256("\n".join(values).encode("utf-8")).hexdigest()


def _json_safe(value: Any) -> Any:
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return value


def _is_absolute_path(value: Any) -> bool:
    return isinstance(value, str) and bool(value) and PurePosixPath(value).is_absolute()


def _expected_evaluations(role: str) -> tuple[str, ...]:
    return EXPECTED_NO_INPUT_EVALUATIONS if role == "head500-noinput" else EXPECTED_FULL_EVALUATIONS


def _selection_val(report: dict[str, Any]) -> dict[str, Any]:
    value = report.get("selection_contract", {}).get("val")
    if not isinstance(value, dict):
        raise RuntimeError("selection_contract.val is missing")
    return value


def _ordered_val_identity(report: dict[str, Any]) -> tuple[list[str], str]:
    selection = _selection_val(report)
    identities = selection.get("sample_identities")
    if not isinstance(identities, list) or not all(isinstance(value, str) for value in identities):
        raise RuntimeError("selection_contract.val.sample_identities is not a string list")
    expected_hash = _sha256_lines(identities)
    reported_hash = selection.get("sample_identity_sha256")
    if len(identities) != EXPECTED_VAL_SAMPLES:
        raise RuntimeError(f"expected val64, found {len(identities)} identities")
    if int(selection.get("sample_count", -1)) != EXPECTED_VAL_SAMPLES:
        raise RuntimeError("selection_contract.val.sample_count is not 64")
    if reported_hash != expected_hash:
        raise RuntimeError("selection_contract.val hash is not the SHA256 of its ordered identities")
    return identities, expected_hash


def _validate_explicit_selection(report: dict[str, Any]) -> None:
    contract = report.get("selection_contract", {})
    explicit = contract.get("explicit_selection")
    if not isinstance(explicit, dict):
        raise RuntimeError("explicit_selection is missing")
    if explicit.get("schema_version") != SELECTION_SCHEMA:
        raise RuntimeError("explicit_selection schema is not Task-3.5b v1")
    if explicit.get("selection_name") != "debiased":
        raise RuntimeError("evaluation did not use the debiased selection")
    if not _is_absolute_path(explicit.get("manifest_path")):
        raise RuntimeError("explicit selection manifest path is not absolute")
    for split in ("train", "val"):
        if explicit.get(split) != contract.get(split):
            raise RuntimeError(f"explicit_selection.{split} differs from verified contract")


def _validate_record_shape(record: dict[str, Any], sample_id: str) -> None:
    if record.get("sample_id") != sample_id:
        raise RuntimeError(f"prediction order mismatch: expected {sample_id!r}, got {record.get('sample_id')!r}")
    required = ("visibility_logits", "gt_visibility", "pred_xy", "gt_xy")
    if any(field not in record for field in required):
        raise RuntimeError(f"prediction record {sample_id!r} lacks compact fields")
    logits = record["visibility_logits"]
    gt_visibility = record["gt_visibility"]
    pred_xy = record["pred_xy"]
    gt_xy = record["gt_xy"]
    if not all(isinstance(value, list) and value for value in (logits, gt_visibility, pred_xy, gt_xy)):
        raise RuntimeError(f"prediction record {sample_id!r} has empty/non-list compact fields")
    if len({len(logits), len(gt_visibility), len(pred_xy), len(gt_xy)}) != 1:
        raise RuntimeError(f"prediction record {sample_id!r} has inconsistent history slots")
    for slot in range(len(logits)):
        views = len(logits[slot])
        if views <= 0 or len(gt_visibility[slot]) != views or len(pred_xy[slot]) != views or len(gt_xy[slot]) != views:
            raise RuntimeError(f"prediction record {sample_id!r} has inconsistent view dimensions")
        for value in logits[slot]:
            if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
                raise RuntimeError(f"prediction record {sample_id!r} has invalid visibility logits")
        for field_name, coordinates in (("pred_xy", pred_xy[slot]), ("gt_xy", gt_xy[slot])):
            for coordinate in coordinates:
                if (
                    not isinstance(coordinate, list)
                    or len(coordinate) != 2
                    or not all(isinstance(value, (int, float)) and math.isfinite(float(value)) for value in coordinate)
                ):
                    raise RuntimeError(f"prediction record {sample_id!r} has invalid {field_name} coordinates")


def prediction_records(
    report: dict[str, Any],
    evaluation: str,
    expected_ids: Sequence[str],
) -> list[dict[str, Any]]:
    metrics = report.get("evaluations", {}).get(evaluation)
    if not isinstance(metrics, dict):
        raise RuntimeError(f"missing evaluation {evaluation!r}")
    records = metrics.get("prediction_records")
    if not isinstance(records, list) or len(records) != EXPECTED_VAL_SAMPLES:
        count = len(records) if isinstance(records, list) else None
        raise RuntimeError(f"evaluation {evaluation!r} has {count} prediction records, expected 64")
    if int(metrics.get("samples", -1)) != EXPECTED_VAL_SAMPLES:
        raise RuntimeError(f"evaluation {evaluation!r} metrics.samples is not 64")
    for record, sample_id in zip(records, expected_ids, strict=True):
        if not isinstance(record, dict):
            raise RuntimeError(f"evaluation {evaluation!r} has a non-object prediction record")
        _validate_record_shape(record, sample_id)
    return records


def _median(values: Sequence[float]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(float(value) for value in values)
    midpoint = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[midpoint]
    return (ordered[midpoint - 1] + ordered[midpoint]) / 2.0


def localization_metrics_from_records(
    records: Sequence[dict[str, Any]],
    *,
    only_history_slot: int | None = None,
) -> dict[str, float | int]:
    oracle_errors: list[float] = []
    u_errors: list[float] = []
    joint_errors: list[float] = []
    view_correct = 0
    visible_histories = 0
    for record in records:
        logits = record["visibility_logits"]
        gt_visibility = record["gt_visibility"]
        pred_xy = record["pred_xy"]
        gt_xy = record["gt_xy"]
        for history_slot, visibility in enumerate(gt_visibility):
            if only_history_slot is not None and history_slot != only_history_slot:
                continue
            positive_views = [index for index, value in enumerate(visibility) if float(value) > 0.5]
            if not positive_views:
                continue
            visible_histories += 1
            selected_view = max(
                range(len(logits[history_slot])),
                key=lambda index: float(logits[history_slot][index]),
            )
            for view in positive_views:
                pred_x, pred_y = map(float, pred_xy[history_slot][view])
                gt_x, gt_y = map(float, gt_xy[history_slot][view])
                oracle_errors.append(math.hypot(pred_x - gt_x, pred_y - gt_y))
                u_errors.append(abs(pred_x - gt_x))
            if selected_view in positive_views:
                view_correct += 1
                pred_x, pred_y = map(float, pred_xy[history_slot][selected_view])
                gt_x, gt_y = map(float, gt_xy[history_slot][selected_view])
                joint_errors.append(math.hypot(pred_x - gt_x, pred_y - gt_y))
            else:
                joint_errors.append(float("inf"))

    def fraction_at(values: Sequence[float], threshold: float) -> float:
        return sum(value <= threshold for value in values) / len(values) if values else float("nan")

    return {
        "visible_view_accuracy": view_correct / visible_histories if visible_histories else float("nan"),
        "visible_history_count": visible_histories,
        "visible_view_count": len(oracle_errors),
        "median_pixel_error": _median(oracle_errors),
        "median_u_error": _median(u_errors),
        "pck4": fraction_at(oracle_errors, 4.0),
        "pck8": fraction_at(oracle_errors, 8.0),
        "joint_median_pixel_error": _median(joint_errors),
        "joint_pck4": fraction_at(joint_errors, 4.0),
        "joint_pck8": fraction_at(joint_errors, 8.0),
    }


def _metric_equal(left: Any, right: Any, tolerance: float = 1e-8) -> bool:
    if isinstance(left, bool) or isinstance(right, bool):
        return left == right
    try:
        left_value = float(left)
        right_value = float(right)
    except (TypeError, ValueError):
        return False
    if math.isnan(left_value) or math.isnan(right_value):
        return math.isnan(left_value) and math.isnan(right_value)
    if math.isinf(left_value) or math.isinf(right_value):
        return left_value == right_value
    return math.isclose(left_value, right_value, rel_tol=tolerance, abs_tol=tolerance)


def _validate_recomputed_metrics(report: dict[str, Any], role: str, identities: Sequence[str]) -> None:
    for evaluation in _expected_evaluations(role):
        reported = report["evaluations"][evaluation]
        recomputed = localization_metrics_from_records(prediction_records(report, evaluation, identities))
        mismatches = [
            metric for metric in RECOMPUTED_METRICS if not _metric_equal(reported.get(metric), recomputed[metric])
        ]
        if mismatches:
            raise RuntimeError(f"{role}/{evaluation} compact predictions disagree with metrics: {mismatches}")


def _ground_truth_projection(records: Sequence[dict[str, Any]]) -> list[tuple[Any, Any]]:
    return [(record["gt_visibility"], record["gt_xy"]) for record in records]


def _validate_ground_truth_pairing(
    reports: dict[str, dict[str, Any]],
    identities: Sequence[str],
) -> None:
    standard_gt = _ground_truth_projection(prediction_records(reports["step0"], "standard", identities))
    for role, report in reports.items():
        role_standard = prediction_records(report, "standard", identities)
        if _ground_truth_projection(role_standard) != standard_gt:
            raise RuntimeError(f"{role} standard ground truth differs from step0")
        for evaluation in _expected_evaluations(role):
            records = prediction_records(report, evaluation, identities)
            if evaluation != "pose-conflict-shifted-target" and _ground_truth_projection(records) != standard_gt:
                raise RuntimeError(f"{role}/{evaluation} unexpectedly changed ground truth")

    for evaluation in EXPECTED_FULL_EVALUATIONS:
        reference = _ground_truth_projection(prediction_records(reports["step0"], evaluation, identities))
        for role in FULL_ROLES[1:]:
            candidate = _ground_truth_projection(prediction_records(reports[role], evaluation, identities))
            if candidate != reference:
                raise RuntimeError(f"ground truth for {evaluation} differs across reports")


def _path_contains(path: str, *tokens: str) -> bool:
    lowered = path.lower()
    return all(token.lower() in lowered for token in tokens)


def _validate_checkpoint_paths(
    reports: dict[str, dict[str, Any]],
    report_paths: dict[str, str | Path],
) -> None:
    if len({str(Path(path).resolve()) for path in report_paths.values()}) != len(REPORT_ROLES):
        raise RuntimeError("input report paths are not unique")
    checkpoints = {role: str(report.get("checkpoint", "")) for role, report in reports.items()}
    loaded_heads = {role: str(report.get("loaded_head_checkpoint", "")) for role, report in reports.items()}
    emitted_heads = {role: str(report.get("head_checkpoint", "")) for role, report in reports.items()}
    for family, paths in (
        ("checkpoint", checkpoints),
        ("loaded head", loaded_heads),
        ("emitted head", emitted_heads),
    ):
        invalid = [role for role, path in paths.items() if not _is_absolute_path(path)]
        if invalid:
            raise RuntimeError(f"{family} paths are not absolute for roles {invalid}")
    if len(set(emitted_heads.values())) != len(REPORT_ROLES):
        raise RuntimeError("evaluation-emitted head checkpoints are reused")
    if len(set(loaded_heads.values())) != len(REPORT_ROLES):
        raise RuntimeError("loaded head checkpoints are reused across logical roles")
    source_pairs = {(checkpoints[role], loaded_heads[role]) for role in REPORT_ROLES}
    if len(source_pairs) != len(REPORT_ROLES):
        raise RuntimeError("two logical roles evaluate the same checkpoint/head pair")

    for role in ("step0", "b25", "b100", "c100"):
        if checkpoints[role] != loaded_heads[role]:
            raise RuntimeError(f"{role} must load LoRA and head from the same pilot checkpoint")
    if not _path_contains(checkpoints["step0"], "heatmap-lora", "checkpoint_step_000000"):
        raise RuntimeError("step0 checkpoint does not identify branch-B step 0")
    if not _path_contains(checkpoints["b25"], "heatmap-lora", "checkpoint_step_000025"):
        raise RuntimeError("b25 checkpoint does not identify branch-B step 25")
    b100 = checkpoints["b100"]
    c100 = checkpoints["c100"]
    if not _path_contains(b100, "heatmap-lora") or not re.search(r"checkpoint_(?:step_000100|final)\.pth$", b100):
        raise RuntimeError("b100 checkpoint does not identify branch-B step 100/final")
    if not _path_contains(c100, "joint-rehearsal") or not re.search(r"checkpoint_(?:step_000100|final)\.pth$", c100):
        raise RuntimeError("c100 checkpoint does not identify branch-C step 100/final")
    if len({checkpoints[role] for role in ("step0", "b25", "b100", "c100")}) != 4:
        raise RuntimeError("pilot checkpoints are not unique across steps/branches")

    if checkpoints["head500-full"] != checkpoints["head500-noinput"]:
        raise RuntimeError("the matched 500-step heads do not share the frozen backbone checkpoint")
    if loaded_heads["head500-full"] == loaded_heads["head500-noinput"]:
        raise RuntimeError("Full and No-input 500-step heads reuse the same head checkpoint")
    if not _path_contains(loaded_heads["head500-full"], "/full/", "head_final.pth"):
        raise RuntimeError("head500-full loaded path does not identify the Full head")
    if not _path_contains(loaded_heads["head500-noinput"], "/no-input/", "head_final.pth"):
        raise RuntimeError("head500-noinput loaded path does not identify the No-input head")


def _selection_report_identity(selection_report: dict[str, Any]) -> tuple[list[str], str]:
    if selection_report.get("selection_ready_for_diagnostic") is not True:
        raise RuntimeError("selection report is not marked ready for diagnostic use")
    manifest = selection_report.get("val", {}).get("debiased", {}).get("manifest")
    if not isinstance(manifest, dict):
        raise RuntimeError("selection report lacks val.debiased.manifest")
    identities = manifest.get("sample_ids")
    if not isinstance(identities, list) or len(identities) != EXPECTED_VAL_SAMPLES:
        raise RuntimeError("selection report does not contain exact debiased val64 identities")
    if not all(isinstance(value, str) for value in identities):
        raise RuntimeError("selection report val identities are not strings")
    identity_hash = _sha256_lines(identities)
    if manifest.get("sample_identity_sha256") != identity_hash:
        raise RuntimeError("selection report debiased val identity hash is inconsistent")
    if int(manifest.get("sample_count", -1)) != EXPECTED_VAL_SAMPLES:
        raise RuntimeError("selection report debiased val count is not 64")
    return identities, identity_hash


def validate_contract(
    reports: dict[str, dict[str, Any]],
    *,
    report_paths: dict[str, str | Path],
    selection_report: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Return named strict checks and their failure details."""

    checks: dict[str, bool] = {}
    errors: dict[str, str] = {}

    def check(name: str, validation: Callable[[], None]) -> None:
        try:
            validation()
        except Exception as error:  # report every independent contract failure
            checks[name] = False
            errors[name] = str(error)
        else:
            checks[name] = True

    def exact_roles() -> None:
        if set(reports) != set(REPORT_ROLES):
            raise RuntimeError(f"expected roles {REPORT_ROLES}, got {sorted(reports)}")
        if set(report_paths) != set(REPORT_ROLES):
            raise RuntimeError("report path roles differ from report roles")

    check("exact_six_roles", exact_roles)
    if not checks["exact_six_roles"]:
        return {"passed": False, "checks": checks, "errors": errors}

    def modes_and_evaluations() -> None:
        for role, report in reports.items():
            expected_mode = "no-input" if role == "head500-noinput" else "full"
            if report.get("mode") != expected_mode:
                raise RuntimeError(f"{role} mode is not {expected_mode}")
            evaluations = report.get("evaluations")
            if not isinstance(evaluations, dict) or set(evaluations) != set(_expected_evaluations(role)):
                raise RuntimeError(f"{role} evaluation set differs: {sorted(evaluations or {})}")

    check("modes_and_exact_evaluation_sets", modes_and_evaluations)

    def exact_algorithm() -> None:
        wrong = [
            role
            for role, report in reports.items()
            if report.get("selection_contract", {}).get("algorithm") != SELECTION_ALGORITHM
        ]
        if wrong:
            raise RuntimeError(f"wrong selection algorithm for roles {wrong}")

    check("selection_algorithm_exact", exact_algorithm)

    identities: list[str] = []
    identity_hash = ""

    def exact_val_identity() -> None:
        nonlocal identities, identity_hash
        projections = {}
        for role, report in reports.items():
            role_ids, role_hash = _ordered_val_identity(report)
            _validate_explicit_selection(report)
            projections[role] = (role_ids, role_hash)
            if report.get("val_samples") != EXPECTED_VAL_SAMPLES:
                raise RuntimeError(f"{role} val_samples is not 64")
            if report.get("selection_contract", {}).get("scene_disjoint") is not True:
                raise RuntimeError(f"{role} selection is not declared scene-disjoint")
        identities, identity_hash = projections["step0"]
        if any(value != projections["step0"] for value in projections.values()):
            raise RuntimeError("the six reports do not share exact ordered val64 identity/hash")
        if selection_report is not None:
            selection_ids, selection_hash = _selection_report_identity(selection_report)
            if (selection_ids, selection_hash) != (identities, identity_hash):
                raise RuntimeError("selection report identity differs from model evaluations")

    check("exact_ordered_val64_identity_and_hash", exact_val_identity)

    def ordered_predictions() -> None:
        if not identities:
            raise RuntimeError("val identity check failed; prediction pairing cannot be verified")
        for role, report in reports.items():
            for evaluation in _expected_evaluations(role):
                prediction_records(report, evaluation, identities)

    check("every_evaluation_has_ordered_val64_predictions", ordered_predictions)

    def metric_recomputation() -> None:
        if not identities:
            raise RuntimeError("val identity check failed; metrics cannot be recomputed")
        for role, report in reports.items():
            _validate_recomputed_metrics(report, role, identities)

    check("compact_predictions_reproduce_localization_metrics", metric_recomputation)

    def ground_truth_pairing() -> None:
        if not identities:
            raise RuntimeError("val identity check failed; ground truth cannot be paired")
        _validate_ground_truth_pairing(reports, identities)

    check("paired_ground_truth_is_exact", ground_truth_pairing)

    def common_snapshot() -> None:
        fields = ("seed", "config", "data_root", "num_history", "max_clip_id")
        for field in fields:
            values = {json.dumps(report.get(field), sort_keys=True) for report in reports.values()}
            if len(values) != 1:
                raise RuntimeError(f"reports differ on {field}: {values}")
        if next(iter(reports.values())).get("num_history") != 2:
            raise RuntimeError("Task-3.5b model evaluation must use two history slots")

    check("same_data_config_seed_and_history_snapshot", common_snapshot)

    def no_training() -> None:
        invalid = []
        for role, report in reports.items():
            if (
                report.get("evaluation_only") is not True
                or not report.get("loaded_head_checkpoint")
                or report.get("train_log") != []
            ):
                invalid.append(role)
        if invalid:
            raise RuntimeError(f"reports contain training or are not evaluation-only: {invalid}")

    check("evaluation_only_no_training", no_training)

    def frozen_qwen_and_full_lora() -> None:
        invalid = [
            role
            for role, report in reports.items()
            if int(report.get("trainable_qwen_tensors", -1)) != 0
            or int(report.get("load", {}).get("matched_lora_tensors", -1)) != EXPECTED_LORA_TENSORS
        ]
        if invalid:
            raise RuntimeError(f"incomplete LoRA load or trainable Qwen in roles {invalid}")

    check("frozen_qwen_and_all_224_lora_loaded", frozen_qwen_and_full_lora)
    check(
        "checkpoint_paths_role_unique_and_reasonable",
        lambda: _validate_checkpoint_paths(reports, report_paths),
    )

    return {
        "passed": all(checks.values()),
        "checks": checks,
        "errors": errors,
        "val_sample_count": EXPECTED_VAL_SAMPLES if identities else None,
        "val_sample_identity_sha256": identity_hash or None,
    }


def paired_prediction_change(
    baseline_records: Sequence[dict[str, Any]],
    candidate_records: Sequence[dict[str, Any]],
    *,
    logit_tolerance: float = 1e-6,
) -> dict[str, Any]:
    """Describe paired peak/logit changes without treating logits as calibrated."""

    if len(baseline_records) != len(candidate_records) or not baseline_records:
        raise RuntimeError("paired prediction lists have unequal or zero length")
    baseline_ids = [record.get("sample_id") for record in baseline_records]
    candidate_ids = [record.get("sample_id") for record in candidate_records]
    if baseline_ids != candidate_ids:
        raise RuntimeError("paired prediction sample order differs")

    sample_xy_changed = 0
    sample_logits_changed = 0
    sample_selected_view_changed = 0
    sample_any_changed = 0
    sample_substantive_changed = 0
    xy_cells = 0
    xy_cells_changed = 0
    visibility_cells = 0
    visibility_cells_changed = 0
    history_slots = 0
    selected_views_changed = 0
    displacements: list[float] = []

    for baseline, candidate in zip(baseline_records, candidate_records, strict=True):
        baseline_logits = baseline["visibility_logits"]
        candidate_logits = candidate["visibility_logits"]
        baseline_xy = baseline["pred_xy"]
        candidate_xy = candidate["pred_xy"]
        if (
            len(baseline_logits) != len(candidate_logits)
            or len(baseline_xy) != len(candidate_xy)
            or len(baseline_logits) != len(baseline_xy)
        ):
            raise RuntimeError("paired prediction shapes differ in history slots")
        any_xy = False
        any_logits = False
        any_selected = False
        for slot in range(len(baseline_logits)):
            if (
                len(baseline_logits[slot]) != len(candidate_logits[slot])
                or len(baseline_xy[slot]) != len(candidate_xy[slot])
                or len(baseline_logits[slot]) != len(baseline_xy[slot])
            ):
                raise RuntimeError("paired prediction shapes differ in views")
            history_slots += 1
            baseline_selected = max(
                range(len(baseline_logits[slot])),
                key=lambda index: float(baseline_logits[slot][index]),
            )
            candidate_selected = max(
                range(len(candidate_logits[slot])),
                key=lambda index: float(candidate_logits[slot][index]),
            )
            selected_changed = baseline_selected != candidate_selected
            selected_views_changed += int(selected_changed)
            any_selected |= selected_changed
            for view in range(len(baseline_logits[slot])):
                visibility_cells += 1
                logit_changed = (
                    abs(float(candidate_logits[slot][view]) - float(baseline_logits[slot][view])) > logit_tolerance
                )
                visibility_cells_changed += int(logit_changed)
                any_logits |= logit_changed
                baseline_x, baseline_y = map(float, baseline_xy[slot][view])
                candidate_x, candidate_y = map(float, candidate_xy[slot][view])
                displacement = math.hypot(
                    candidate_x - baseline_x,
                    candidate_y - baseline_y,
                )
                displacements.append(displacement)
                xy_cells += 1
                xy_changed = displacement > 0.0
                xy_cells_changed += int(xy_changed)
                any_xy |= xy_changed
        sample_xy_changed += int(any_xy)
        sample_logits_changed += int(any_logits)
        sample_selected_view_changed += int(any_selected)
        sample_any_changed += int(any_xy or any_logits)
        sample_substantive_changed += int(any_xy or any_selected)

    samples = len(baseline_records)
    return {
        "samples": samples,
        "logit_change_tolerance": logit_tolerance,
        "sample_any_pred_xy_changed_fraction": sample_xy_changed / samples,
        "sample_any_visibility_logit_changed_fraction": sample_logits_changed / samples,
        "sample_any_selected_view_changed_fraction": sample_selected_view_changed / samples,
        "sample_any_prediction_changed_fraction": sample_any_changed / samples,
        "sample_any_substantive_prediction_changed_fraction": (sample_substantive_changed / samples),
        "pred_xy_cell_changed_fraction": xy_cells_changed / xy_cells,
        "visibility_logit_cell_changed_fraction": (visibility_cells_changed / visibility_cells),
        "selected_view_changed_fraction": selected_views_changed / history_slots,
        "mean_pred_xy_peak_displacement": sum(displacements) / len(displacements),
        "max_pred_xy_peak_displacement": max(displacements),
    }


def _numeric_metric_deltas(
    candidate: dict[str, Any],
    baseline: dict[str, Any],
) -> dict[str, float | None]:
    result: dict[str, float | None] = {}
    for metric in SUMMARY_METRICS:
        left = candidate.get(metric)
        right = baseline.get(metric)
        if isinstance(left, (int, float)) and isinstance(right, (int, float)):
            left_value = float(left)
            right_value = float(right)
            result[metric] = (
                left_value - right_value if math.isfinite(left_value) and math.isfinite(right_value) else None
            )
        else:
            result[metric] = None
    return result


def _localization_degradation(
    standard: dict[str, Any],
    intervention: dict[str, Any],
) -> dict[str, float | None]:
    standard_median = float(standard["median_pixel_error"])
    intervention_median = float(intervention["median_pixel_error"])
    median_relative_increase = None
    if math.isfinite(standard_median) and standard_median > 0.0 and math.isfinite(intervention_median):
        median_relative_increase = (intervention_median - standard_median) / standard_median
    return {
        "pck8_drop": float(standard["pck8"]) - float(intervention["pck8"]),
        "joint_pck8_drop": float(standard["joint_pck8"]) - float(intervention["joint_pck8"]),
        "visible_view_accuracy_drop": float(standard["visible_view_accuracy"])
        - float(intervention["visible_view_accuracy"]),
        "median_pixel_error_increase": (
            intervention_median - standard_median
            if math.isfinite(intervention_median) and math.isfinite(standard_median)
            else None
        ),
        "median_pixel_error_relative_increase": median_relative_increase,
    }


def _binomial_greater_pvalue(successes: int, trials: int, null_probability: float) -> float:
    if trials <= 0 or successes < 0 or successes > trials:
        raise ValueError("invalid binomial counts")
    if not 0.0 <= null_probability <= 1.0:
        raise ValueError("null probability must lie in [0,1]")
    if null_probability == 0.0:
        return 0.0 if successes > 0 else 1.0
    if null_probability == 1.0:
        return 1.0
    return min(
        1.0,
        sum(
            math.comb(trials, count) * (null_probability**count) * ((1.0 - null_probability) ** (trials - count))
            for count in range(successes, trials + 1)
        ),
    )


def _scene_from_sample_id(sample_id: str) -> str:
    clip_identity, separator, _frame = sample_id.rpartition(":frame=")
    if not separator:
        raise RuntimeError(f"sample identity lacks ':frame=' suffix: {sample_id!r}")
    scene = PurePosixPath(clip_identity).parent.name
    if not scene:
        raise RuntimeError(f"cannot extract scene from sample identity: {sample_id!r}")
    return scene


def _joint_success_counts(record: dict[str, Any]) -> tuple[int, int]:
    successes = 0
    trials = 0
    for slot, visibility in enumerate(record["gt_visibility"]):
        positive_views = [index for index, value in enumerate(visibility) if float(value) > 0.5]
        if not positive_views:
            continue
        trials += 1
        selected_view = max(
            range(len(record["visibility_logits"][slot])),
            key=lambda index: float(record["visibility_logits"][slot][index]),
        )
        if selected_view not in positive_views:
            continue
        pred_x, pred_y = map(float, record["pred_xy"][slot][selected_view])
        gt_x, gt_y = map(float, record["gt_xy"][slot][selected_view])
        successes += int(math.hypot(pred_x - gt_x, pred_y - gt_y) <= 8.0)
    return successes, trials


def _scene_cluster_lower_bound(
    records: Sequence[dict[str, Any]],
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    clusters: dict[str, list[int]] = {}
    for record in records:
        scene = _scene_from_sample_id(str(record["sample_id"]))
        successes, trials = _joint_success_counts(record)
        counts = clusters.setdefault(scene, [0, 0])
        counts[0] += successes
        counts[1] += trials
    if len(clusters) < 2:
        raise RuntimeError("scene-cluster significance requires at least two held-out scenes")
    scenes = sorted(clusters)
    rng = random.Random(seed)
    distribution: list[float] = []
    for _ in range(replicates):
        successes = 0
        trials = 0
        for _draw in scenes:
            selected = scenes[rng.randrange(len(scenes))]
            successes += clusters[selected][0]
            trials += clusters[selected][1]
        if trials:
            distribution.append(successes / trials)
    if len(distribution) != replicates:
        raise RuntimeError("scene-cluster bootstrap produced an empty-trial replicate")
    distribution.sort()

    def percentile(probability: float) -> float:
        position = probability * (len(distribution) - 1)
        lower = math.floor(position)
        upper = math.ceil(position)
        if lower == upper:
            return distribution[lower]
        fraction = position - lower
        return distribution[lower] * (1.0 - fraction) + distribution[upper] * fraction

    return {
        "method": "scene_cluster_percentile_bootstrap",
        "resampling_unit": "scene",
        "scene_count": len(scenes),
        "scene_counts": {scene: {"successes": clusters[scene][0], "trials": clusters[scene][1]} for scene in scenes},
        "replicates": replicates,
        "seed": seed,
        "ci95": [percentile(0.025), percentile(0.975)],
    }


def _significance_against_prior(
    metrics: dict[str, Any],
    records: Sequence[dict[str, Any]],
    prior: float,
    *,
    seed: int,
) -> dict[str, Any]:
    value = float(metrics["joint_pck8"])
    trials = int(metrics["visible_history_count"])
    successes = round(value * trials)
    reconstructed = successes / trials if trials else float("nan")
    consistent = trials > 0 and math.isclose(
        value,
        reconstructed,
        rel_tol=1e-8,
        abs_tol=max(1e-8, 0.5 / max(trials, 1)),
    )
    pvalue = _binomial_greater_pvalue(successes, trials, prior) if consistent else 1.0
    cluster_bootstrap = _scene_cluster_lower_bound(
        records,
        replicates=CLUSTER_BOOTSTRAP_REPLICATES,
        seed=seed,
    )
    absolute_delta = value - prior
    checks = {
        "observed_above_prior": value > prior,
        "absolute_delta_at_least_0_10": absolute_delta >= MIN_PRIOR_JOINT_PCK8_DELTA,
        "one_sided_exact_binomial_p_below_alpha": pvalue < SIGNIFICANCE_ALPHA,
        "scene_cluster_ci_lower_bound_above_prior": float(cluster_bootstrap["ci95"][0]) > prior,
        "reported_fraction_consistent_with_counts": consistent,
    }
    return {
        "test": "effect_size_plus_exact_binomial_and_scene_cluster_bootstrap",
        "null_joint_pck8": prior,
        "observed_joint_pck8": value,
        "absolute_delta": absolute_delta,
        "minimum_absolute_delta": MIN_PRIOR_JOINT_PCK8_DELTA,
        "successes": successes,
        "trials": trials,
        "reported_fraction_consistent_with_counts": consistent,
        "one_sided_exact_binomial_pvalue": pvalue,
        "scene_cluster_bootstrap": cluster_bootstrap,
        "alpha": SIGNIFICANCE_ALPHA,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _resolve_empirical_prior(
    selection_report: dict[str, Any] | None,
    explicit_value: float | None,
) -> tuple[float, str]:
    if selection_report is not None:
        value = (
            selection_report.get("empirical_prior_strength", {})
            .get("after_debiased", {})
            .get("metrics", {})
            .get("joint_pck8")
        )
        if not isinstance(value, (int, float)) or not math.isfinite(float(value)):
            raise RuntimeError(
                "selection report lacks finite empirical_prior_strength.after_debiased.metrics.joint_pck8"
            )
        resolved = float(value)
        if explicit_value is not None and not math.isclose(resolved, explicit_value, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError("explicit empirical prior disagrees with the passed selection report")
        return resolved, "selection_report.after_debiased"
    resolved = DEFAULT_EMPIRICAL_PRIOR_JOINT_PCK8 if explicit_value is None else float(explicit_value)
    if not 0.0 <= resolved <= 1.0:
        raise ValueError("empirical-prior joint PCK8 must lie in [0,1]")
    source = "recorded_default" if explicit_value is None else "cli"
    return resolved, source


def _view_support_from_selection(
    selection_report: dict[str, Any] | None,
) -> dict[str, Any]:
    if selection_report is None:
        return {
            "available": False,
            "recent_slot": None,
            "recent_slot_only_back": None,
            "candidate_slot_by_view_complete": False,
            "selected_slot_by_view_complete": False,
            "slot_by_view_complete": False,
            "reason": "selection report not supplied; slot-by-view completeness is unverified",
        }

    summaries: dict[str, Any] = {}
    for family, audit in (
        (
            "candidate",
            selection_report.get("val", {}).get("candidate_catalog", {}).get("audit", {}),
        ),
        (
            "selected",
            selection_report.get("val", {}).get("debiased", {}).get("audit", {}),
        ),
    ):
        slots = audit.get("per_history_slot") if isinstance(audit, dict) else None
        if not isinstance(slots, dict) or not slots:
            raise RuntimeError(f"selection report lacks val {family} per-history-slot audit")
        normalized: dict[str, Any] = {}
        for key, value in slots.items():
            views = sorted(view for view, count in value.get("view_counts", {}).items() if int(count) > 0)
            normalized[str(int(key))] = {
                "positive_views": views,
                "view_counts": value.get("view_counts", {}),
                "all_four_views_supported": set(views) == set(VIEW_NAMES),
            }
        summaries[family] = normalized

    slot_keys = sorted(set(summaries["candidate"]) | set(summaries["selected"]), key=int)
    recent_slot = slot_keys[-1]
    candidate_complete = all(value["all_four_views_supported"] for value in summaries["candidate"].values())
    selected_complete = all(value["all_four_views_supported"] for value in summaries["selected"].values())
    candidate_recent_views = summaries["candidate"].get(recent_slot, {}).get("positive_views", [])
    selected_recent_views = summaries["selected"].get(recent_slot, {}).get("positive_views", [])
    recent_only_back = bool(set(candidate_recent_views) == {"back"} or set(selected_recent_views) == {"back"})
    return {
        "available": True,
        "recent_slot": int(recent_slot),
        "recent_slot_only_back": recent_only_back,
        "candidate_slot_by_view_complete": candidate_complete,
        "selected_slot_by_view_complete": selected_complete,
        "slot_by_view_complete": candidate_complete and selected_complete,
        "families": summaries,
        "interpretation": (
            "View support is a data-structure property, not a model effect. "
            "A recent slot with only back-view positives prevents a complete "
            "slot-by-view grounding claim even if aggregate interventions pass."
        ),
    }


def _build_rows(reports: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for role in REPORT_ROLES:
        report = reports[role]
        standard = report["evaluations"]["standard"]
        for evaluation in _expected_evaluations(role):
            metrics = report["evaluations"][evaluation]
            deltas = _numeric_metric_deltas(metrics, standard)
            row: dict[str, Any] = {
                "role": role,
                "mode": report["mode"],
                "evaluation": evaluation,
                "is_standard": evaluation == "standard",
            }
            row.update({metric: metrics.get(metric) for metric in SUMMARY_METRICS})
            row.update({f"delta_vs_standard_{metric}": value for metric, value in deltas.items()})
            rows.append(row)
    return rows


def _build_intervention_summaries(
    reports: dict[str, dict[str, Any]],
    identities: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    intervention_deltas: dict[str, Any] = {}
    paired: dict[str, Any] = {}
    for role in FULL_ROLES:
        report = reports[role]
        standard = report["evaluations"]["standard"]
        standard_records = prediction_records(report, "standard", identities)
        intervention_deltas[role] = {}
        paired[role] = {}
        for evaluation in EXPECTED_FULL_EVALUATIONS[1:]:
            metrics = report["evaluations"][evaluation]
            intervention_deltas[role][evaluation] = {
                "metric_delta_intervention_minus_standard": _numeric_metric_deltas(metrics, standard),
                "localization_degradation": _localization_degradation(standard, metrics),
            }
            paired[role][evaluation] = paired_prediction_change(
                standard_records,
                prediction_records(report, evaluation, identities),
            )
    return intervention_deltas, paired


def _build_cross_model_comparisons(
    reports: dict[str, dict[str, Any]],
    identities: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    definitions = {
        "head500-full_vs_head500-noinput": ("head500-noinput", "head500-full"),
        "b25_vs_step0": ("step0", "b25"),
        "b100_vs_step0": ("step0", "b100"),
        "c100_vs_step0": ("step0", "c100"),
    }
    metric_comparisons: dict[str, Any] = {}
    paired_comparisons: dict[str, Any] = {}
    for name, (baseline_role, candidate_role) in definitions.items():
        baseline = reports[baseline_role]["evaluations"]["standard"]
        candidate = reports[candidate_role]["evaluations"]["standard"]
        metric_comparisons[name.replace("_vs_", "_minus_")] = _numeric_metric_deltas(candidate, baseline)
        paired_comparisons[name] = paired_prediction_change(
            prediction_records(reports[baseline_role], "standard", identities),
            prediction_records(reports[candidate_role], "standard", identities),
        )
    return metric_comparisons, paired_comparisons


def _build_per_history_slot_metrics(
    reports: dict[str, dict[str, Any]],
    identities: Sequence[str],
) -> tuple[dict[str, Any], dict[str, Any]]:
    reference = prediction_records(reports["step0"], "standard", identities)
    history_slots = len(reference[0]["gt_visibility"])
    if history_slots <= 0:
        raise RuntimeError("Task-3.5b predictions contain no history slots")

    result: dict[str, Any] = {}
    for role, report in reports.items():
        result[role] = {}
        for evaluation in _expected_evaluations(role):
            records = prediction_records(report, evaluation, identities)
            result[role][evaluation] = {
                str(slot): localization_metrics_from_records(
                    records,
                    only_history_slot=slot,
                )
                for slot in range(history_slots)
            }

    recent_slot = history_slots - 1
    audit: dict[str, Any] = {
        "recent_slot": recent_slot,
        "models": {},
    }
    for role in ("head500-full", "b100", "c100"):
        slot_metrics = result[role]
        standard_recent = slot_metrics["standard"][str(recent_slot)]
        blank_recent = slot_metrics["blank-images"][str(recent_slot)]
        history_recent = slot_metrics["history-shuffle"][str(recent_slot)]
        current_recent = slot_metrics["current-shuffle"][str(recent_slot)]
        standard_oldest = slot_metrics["standard"]["0"]
        audit["models"][role] = {
            "oldest_slot_standard": standard_oldest,
            "recent_slot_standard": standard_recent,
            "recent_slot_blank_images": blank_recent,
            "recent_slot_history_shuffle": history_recent,
            "recent_slot_current_shuffle": current_recent,
            "recent_view_accuracy_stays_perfect_when_blank": bool(
                math.isclose(float(standard_recent["visible_view_accuracy"]), 1.0)
                and math.isclose(float(blank_recent["visible_view_accuracy"]), 1.0)
            ),
            "recent_history_shuffle_joint_pck8_delta": float(
                history_recent["joint_pck8"]
            )
            - float(standard_recent["joint_pck8"]),
            "recent_current_shuffle_joint_pck8_delta": float(
                current_recent["joint_pck8"]
            )
            - float(standard_recent["joint_pck8"]),
        }
    audit["interpretation"] = (
        "This is descriptive slot-wise evidence. A perfect recent-slot view "
        "accuracy that survives blank images is compatible with the structural "
        "back-view prior, not proof of visual view selection. Low oldest-slot "
        "joint PCK8 and shuffle invariance indicate that aggregate scores can be "
        "dominated by the recent slot."
    )
    return result, audit


def _visual_grounding_gate(
    reports: dict[str, dict[str, Any]],
    within_paired: dict[str, Any],
    empirical_prior: float,
) -> dict[str, Any]:
    models: dict[str, Any] = {}
    for role in ("head500-full", "b100", "c100"):
        evaluations = reports[role]["evaluations"]
        standard = evaluations["standard"]
        blank = evaluations["blank-images"]
        significance = _significance_against_prior(
            standard,
            standard["prediction_records"],
            empirical_prior,
            seed=3500 + list(("head500-full", "b100", "c100")).index(role),
        )
        blank_degradation = _localization_degradation(standard, blank)
        blank_pair = within_paired[role]["blank-images"]
        blank_checks = {
            "joint_pck8_at_or_below_empirical_prior": float(blank["joint_pck8"]) <= empirical_prior,
            "joint_pck8_drop_material": float(blank_degradation["joint_pck8_drop"]) >= MIN_BLANK_JOINT_PCK8_DROP,
            "paired_prediction_change_material": float(blank_pair["sample_any_substantive_prediction_changed_fraction"])
            >= MIN_BLANK_PAIRED_CHANGE_FRACTION,
        }
        blank_passed = all(blank_checks.values())

        shuffle_results: dict[str, Any] = {}
        for intervention in ("history-shuffle", "current-shuffle"):
            degradation = _localization_degradation(standard, evaluations[intervention])
            pair = within_paired[role][intervention]
            localization_material = bool(
                float(degradation["joint_pck8_drop"]) >= MIN_SHUFFLE_LOCALIZATION_DROP
                or float(degradation["pck8_drop"]) >= MIN_SHUFFLE_LOCALIZATION_DROP
                or (
                    degradation["median_pixel_error_relative_increase"] is not None
                    and float(degradation["median_pixel_error_relative_increase"]) >= 0.20
                )
            )
            paired_material = bool(
                float(pair["sample_any_substantive_prediction_changed_fraction"]) >= MIN_SHUFFLE_PAIRED_CHANGE_FRACTION
            )
            shuffle_results[intervention] = {
                "localization_degradation": degradation,
                "paired_prediction_change": pair,
                "localization_change_material": localization_material,
                "paired_prediction_change_material": paired_material,
                "passed": localization_material and paired_material,
            }
        at_least_one_shuffle = any(value["passed"] for value in shuffle_results.values())
        checks = {
            "standard_significantly_exceeds_empirical_prior": significance["passed"],
            "blank_localization_collapses": blank_passed,
            "history_or_current_shuffle_is_material": at_least_one_shuffle,
        }
        models[role] = {
            "checks": checks,
            "passed": all(checks.values()),
            "significance": significance,
            "blank": {
                "checks": blank_checks,
                "passed": blank_passed,
                "localization_degradation": blank_degradation,
                "paired_prediction_change": blank_pair,
            },
            "shuffles": shuffle_results,
        }

    return {
        "passed": all(model["passed"] for model in models.values()),
        "models": models,
        "thresholds": {
            "empirical_prior_joint_pck8": empirical_prior,
            "minimum_prior_joint_pck8_absolute_delta": MIN_PRIOR_JOINT_PCK8_DELTA,
            "one_sided_significance_alpha": SIGNIFICANCE_ALPHA,
            "scene_cluster_bootstrap_replicates": CLUSTER_BOOTSTRAP_REPLICATES,
            "blank_min_joint_pck8_drop": MIN_BLANK_JOINT_PCK8_DROP,
            "blank_min_paired_substantive_change_fraction": (MIN_BLANK_PAIRED_CHANGE_FRACTION),
            "shuffle_min_pck8_or_joint_pck8_drop": MIN_SHUFFLE_LOCALIZATION_DROP,
            "shuffle_min_median_relative_increase": 0.20,
            "shuffle_min_paired_substantive_change_fraction": (MIN_SHUFFLE_PAIRED_CHANGE_FRACTION),
        },
        "gate_definition": (
            "Each of head500-Full, B100, and C100 must significantly beat the "
            "empirical-prior joint-PCK8 null; blank images must collapse below "
            "the null with a material paired change; and at least one of history "
            "or current shuffle must cause both material localization degradation "
            "and material paired peak/selected-view changes."
        ),
    }


def build_summary(
    reports: dict[str, dict[str, Any]],
    *,
    report_paths: dict[str, str | Path],
    selection_report: dict[str, Any] | None = None,
    empirical_prior_joint_pck8: float | None = None,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    contract = validate_contract(
        reports,
        report_paths=report_paths,
        selection_report=selection_report,
    )
    if not contract["passed"]:
        raise RuntimeError(f"Task-3.5b debiased model-eval contract failed: {contract['errors']}")
    identities, _identity_hash = _ordered_val_identity(reports["step0"])
    empirical_prior, prior_source = _resolve_empirical_prior(
        selection_report,
        empirical_prior_joint_pck8,
    )
    rows = _build_rows(reports)
    intervention_deltas, within_paired = _build_intervention_summaries(reports, identities)
    standard_comparisons, cross_paired = _build_cross_model_comparisons(reports, identities)
    per_history_slot, recent_slot_audit = _build_per_history_slot_metrics(
        reports,
        identities,
    )
    gate = _visual_grounding_gate(reports, within_paired, empirical_prior)
    view_support = _view_support_from_selection(selection_report)
    slot_complete = bool(view_support["slot_by_view_complete"])
    aggregate_supported = bool(gate["passed"])
    complete_grounding = aggregate_supported and slot_complete
    recent_only_back = view_support["recent_slot_only_back"]
    if aggregate_supported and not complete_grounding:
        claim_scope = "aggregate_only_not_complete_slot_by_view_grounding"
    elif complete_grounding:
        claim_scope = "aggregate_and_slot_by_view_support_but_not_causal_slotwise_proof"
    else:
        claim_scope = "aggregate_visual_grounding_not_established"

    summary = {
        "task": "task35b_debiased_model_eval_v1",
        "contract": contract,
        "empirical_prior": {
            "joint_pck8": empirical_prior,
            "source": prior_source,
        },
        "standard_metrics": {
            role: {metric: reports[role]["evaluations"]["standard"].get(metric) for metric in SUMMARY_METRICS}
            for role in REPORT_ROLES
        },
        "standard_comparisons": standard_comparisons,
        "per_history_slot_metrics": per_history_slot,
        "recent_slot_shortcut_audit": recent_slot_audit,
        "intervention_deltas": intervention_deltas,
        "paired_prediction_changes": {
            "within_report_interventions": within_paired,
            "cross_model_standard": cross_paired,
            "definition": (
                "Fractions are paired over the exact ordered val64. Logit-value "
                "changes use absolute tolerance 1e-6; substantive changes require "
                "a heatmap argmax coordinate or visibility-selected view to change."
            ),
        },
        "visual_grounding_gate": gate,
        "slot_view_support": view_support,
        "conclusion": {
            "aggregate_shortcut_suppression_supported": aggregate_supported,
            "slot_view_support_complete": slot_complete,
            "slot_view_grounding_complete": complete_grounding,
            "recent_slot_only_back": recent_only_back,
            "claim_scope": claim_scope,
            "interpretation": (
                "The intervention gate addresses aggregate dependence on visual "
                "inputs. It does not repair or prove unsupported slot-by-view cells; "
                "recent-slot-only-back support must remain an explicit limitation."
            ),
        },
        "reports": {role: str(Path(report_paths[role]).resolve()) for role in REPORT_ROLES},
    }
    return summary, rows


def main() -> int:
    args = parse_args()
    report_paths = {role: Path(getattr(args, f"{role.replace('-', '_')}_report")) for role in REPORT_ROLES}
    reports = {role: load_json(path) for role, path in report_paths.items()}
    selection_report = load_json(args.selection_report) if args.selection_report else None
    summary, rows = build_summary(
        reports,
        report_paths=report_paths,
        selection_report=selection_report,
        empirical_prior_joint_pck8=args.empirical_prior_joint_pck8,
    )

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "task35b_debiased_model_summary.json"
    csv_path = output_dir / "task35b_debiased_model_summary.csv"
    with json_path.open("w", encoding="utf-8") as handle:
        json.dump(
            _json_safe(summary),
            handle,
            indent=2,
            ensure_ascii=False,
            allow_nan=False,
        )
    fieldnames = list(rows[0])
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(_json_safe(rows))
    print(json.dumps(_json_safe(summary), indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

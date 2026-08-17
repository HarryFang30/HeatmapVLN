from __future__ import annotations

import hashlib
import json
import math
import sys
from copy import deepcopy
from pathlib import Path

import pytest
from scripts.tools.summarize_task35b_debiased_model_eval import (
    DEFAULT_EMPIRICAL_PRIOR_JOINT_PCK8,
    EXPECTED_FULL_EVALUATIONS,
    REPORT_ROLES,
    build_summary,
    main,
    paired_prediction_change,
    validate_contract,
)


def _identities() -> list[str]:
    return [f"val/scene_{index % 8}/clip_{index:03d}:frame=12" for index in range(64)]


def _selection_contract() -> dict:
    identities = _identities()
    val = {
        "sample_count": 64,
        "sample_identity_sha256": hashlib.sha256("\n".join(identities).encode("utf-8")).hexdigest(),
        "sample_identities": identities,
        "scenes": sorted({identity.split("/")[1] for identity in identities}),
    }
    train = {
        "sample_count": 128,
        "sample_identity_sha256": "a" * 64,
        "sample_identities": [f"train/sample_{index}" for index in range(128)],
        "scenes": ["train_scene"],
    }
    return {
        "algorithm": "task35b_verified_explicit_manifest_v1",
        "scene_disjoint": True,
        "explicit_selection": {
            "schema_version": "task35b_debiased_selection_v1",
            "selection_name": "debiased",
            "manifest_path": "/selection/selection_manifest.json",
            "train": deepcopy(train),
            "val": deepcopy(val),
        },
        "train": train,
        "val": val,
    }


def _records(successful_samples: int, *, logit_bias: float = 0.0) -> list[dict]:
    records = []
    for sample_index, sample_id in enumerate(_identities()):
        succeeds = sample_index < successful_samples
        gt_visibility = [[1.0, 0.0, 0.0, 0.0], [0.0, 0.0, 1.0, 0.0]]
        visibility_logits = (
            [[4.0 + logit_bias, 0.0, -1.0, -2.0], [-2.0, -1.0, 4.0 + logit_bias, 0.0]]
            if succeeds
            else [[0.0, 4.0 + logit_bias, -1.0, -2.0], [-2.0, -1.0, 0.0, 4.0 + logit_bias]]
        )
        gt_xy = [[[10, 10] for _ in range(4)] for _ in range(2)]
        pred_xy = deepcopy(gt_xy)
        if not succeeds:
            pred_xy[0][0] = [30, 30]
            pred_xy[1][2] = [30, 30]
        records.append(
            {
                "sample_id": sample_id,
                "visibility_logits": visibility_logits,
                "gt_visibility": gt_visibility,
                "pred_xy": pred_xy,
                "gt_xy": gt_xy,
            }
        )
    return records


def _median(values: list[float]) -> float:
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[middle]
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _metrics(records: list[dict]) -> dict:
    oracle_errors: list[float] = []
    u_errors: list[float] = []
    joint_errors: list[float] = []
    view_correct = 0
    visible_histories = 0
    for record in records:
        for slot, gt_views in enumerate(record["gt_visibility"]):
            positive_views = [index for index, value in enumerate(gt_views) if value > 0.5]
            if not positive_views:
                continue
            visible_histories += 1
            selected = max(
                range(len(record["visibility_logits"][slot])),
                key=lambda index: record["visibility_logits"][slot][index],
            )
            for view in positive_views:
                pred_x, pred_y = record["pred_xy"][slot][view]
                gt_x, gt_y = record["gt_xy"][slot][view]
                oracle_errors.append(math.hypot(pred_x - gt_x, pred_y - gt_y))
                u_errors.append(abs(pred_x - gt_x))
            if selected in positive_views:
                view_correct += 1
                pred_x, pred_y = record["pred_xy"][slot][selected]
                gt_x, gt_y = record["gt_xy"][slot][selected]
                joint_errors.append(math.hypot(pred_x - gt_x, pred_y - gt_y))
            else:
                joint_errors.append(float("inf"))
    return {
        "loss": 1.0,
        "visibility_auroc": 0.8,
        "visibility_auprc": 0.8,
        "visibility_f1": 0.8,
        "visibility_precision": 0.8,
        "visibility_recall": 0.8,
        "visible_view_accuracy": view_correct / visible_histories,
        "visible_history_count": visible_histories,
        "visible_view_count": len(oracle_errors),
        "median_pixel_error": _median(oracle_errors),
        "median_u_error": _median(u_errors),
        "pck4": sum(error <= 4.0 for error in oracle_errors) / len(oracle_errors),
        "pck8": sum(error <= 8.0 for error in oracle_errors) / len(oracle_errors),
        "joint_median_pixel_error": _median(joint_errors),
        "joint_pck4": sum(error <= 4.0 for error in joint_errors) / len(joint_errors),
        "joint_pck8": sum(error <= 8.0 for error in joint_errors) / len(joint_errors),
        "samples": 64,
        "prediction_records": records,
    }


def _evaluation(successful_samples: int, *, logit_bias: float = 0.0) -> dict:
    return _metrics(_records(successful_samples, logit_bias=logit_bias))


def _paths(role: str) -> tuple[str, str]:
    if role == "step0":
        path = "/pilot/heatmap-lora/checkpoint_step_000000.pth"
        return path, path
    if role == "b25":
        path = "/pilot/heatmap-lora/checkpoint_step_000025.pth"
        return path, path
    if role == "b100":
        path = "/pilot/heatmap-lora/checkpoint_final.pth"
        return path, path
    if role == "c100":
        path = "/pilot/joint-rehearsal/checkpoint_final.pth"
        return path, path
    mode = "full" if role == "head500-full" else "no-input"
    return "/base/latest.pth", f"/task35/{mode}/head_final.pth"


def _report(role: str) -> dict:
    success_by_role = {
        "step0": 20,
        "b25": 40,
        "b100": 64,
        "c100": 56,
        "head500-full": 64,
        "head500-noinput": 16,
    }
    checkpoint, loaded_head = _paths(role)
    mode = "no-input" if role == "head500-noinput" else "full"
    evaluations = {"standard": _evaluation(success_by_role[role])}
    if mode == "full":
        evaluations.update(
            {
                "zero-pose": _evaluation(success_by_role[role], logit_bias=0.25),
                "blank-images": _evaluation(0, logit_bias=0.5),
                "history-shuffle": _evaluation(24, logit_bias=0.75),
                "current-shuffle": _evaluation(32, logit_bias=1.0),
                "pose-conflict": _evaluation(28, logit_bias=1.25),
                "pose-conflict-shifted-target": _evaluation(28, logit_bias=1.5),
            }
        )
    return {
        "mode": mode,
        "seed": 42,
        "config": "/config.yaml",
        "checkpoint": checkpoint,
        "data_root": "/randomwalk",
        "load": {"matched_lora_tensors": 224},
        "initial_head_hash": f"initial-{role}",
        "train_steps": 500,
        "evaluation_only": True,
        "loaded_head_checkpoint": loaded_head,
        "train_samples": 128,
        "val_samples": 64,
        "selection_contract": _selection_contract(),
        "num_history": 2,
        "max_clip_id": 2000,
        "trainable_qwen_tensors": 0,
        "train_log": [],
        "evaluations": evaluations,
        "head_checkpoint": f"/eval/{role}/head_final.pth",
    }


def _reports() -> dict[str, dict]:
    return {role: _report(role) for role in REPORT_ROLES}


def _report_paths(tmp_path: Path) -> dict[str, Path]:
    return {role: tmp_path / role / "report.json" for role in REPORT_ROLES}


def _selection_report() -> dict:
    identities = _identities()
    val_manifest = {
        "sample_count": 64,
        "sample_ids": identities,
        "sample_identity_sha256": hashlib.sha256("\n".join(identities).encode("utf-8")).hexdigest(),
    }
    audit = {
        "per_history_slot": {
            "0": {"view_counts": {"front": 20, "right": 10, "back": 12, "left": 11}},
            "1": {"view_counts": {"back": 64}},
        }
    }
    return {
        "selection_ready_for_diagnostic": True,
        "val": {
            "candidate_catalog": {"audit": deepcopy(audit)},
            "debiased": {"manifest": val_manifest, "audit": deepcopy(audit)},
        },
        "empirical_prior_strength": {
            "after_debiased": {
                "available": True,
                "metrics": {"joint_pck8": DEFAULT_EMPIRICAL_PRIOR_JOINT_PCK8},
            }
        },
    }


def test_build_summary_validates_exact_pairing_and_separates_claim_scope(tmp_path):
    reports = _reports()
    summary, rows = build_summary(
        reports,
        report_paths=_report_paths(tmp_path),
        selection_report=_selection_report(),
    )

    assert summary["contract"]["passed"] is True
    assert len(rows) == 5 * len(EXPECTED_FULL_EVALUATIONS) + 1
    assert summary["visual_grounding_gate"]["passed"] is True
    assert set(summary["visual_grounding_gate"]["models"]) == {
        "head500-full",
        "b100",
        "c100",
    }
    assert summary["standard_comparisons"]["b100_minus_step0"]["joint_pck8"] > 0.0
    assert summary["per_history_slot_metrics"]["b100"]["standard"]["0"]["visible_history_count"] == 64
    assert summary["recent_slot_shortcut_audit"]["recent_slot"] == 1
    assert (
        summary["recent_slot_shortcut_audit"]["models"]["b100"]
        ["recent_view_accuracy_stays_perfect_when_blank"]
        is False
    )
    head_comparison = summary["paired_prediction_changes"]["cross_model_standard"]["head500-full_vs_head500-noinput"]
    assert head_comparison["sample_any_prediction_changed_fraction"] > 0.0
    conclusion = summary["conclusion"]
    assert conclusion["aggregate_shortcut_suppression_supported"] is True
    assert conclusion["slot_view_grounding_complete"] is False
    assert conclusion["recent_slot_only_back"] is True
    assert conclusion["claim_scope"] == "aggregate_only_not_complete_slot_by_view_grounding"


def test_contract_rejects_wrong_algorithm_training_and_reused_checkpoint(tmp_path):
    reports = _reports()
    reports["b25"]["selection_contract"]["algorithm"] = "wrong"
    reports["c100"]["evaluation_only"] = False
    reports["c100"]["train_log"] = [{"step": 1}]
    reports["b100"]["checkpoint"] = reports["b25"]["checkpoint"]
    reports["b100"]["loaded_head_checkpoint"] = reports["b25"]["loaded_head_checkpoint"]

    contract = validate_contract(
        reports,
        report_paths=_report_paths(tmp_path),
        selection_report=_selection_report(),
    )

    assert contract["passed"] is False
    assert contract["checks"]["selection_algorithm_exact"] is False
    assert contract["checks"]["evaluation_only_no_training"] is False
    assert contract["checks"]["checkpoint_paths_role_unique_and_reasonable"] is False
    with pytest.raises(RuntimeError, match="contract failed"):
        build_summary(
            reports,
            report_paths=_report_paths(tmp_path),
            selection_report=_selection_report(),
        )


def test_contract_rejects_prediction_count_or_order_mismatch(tmp_path):
    reports = _reports()
    reports["b25"]["evaluations"]["current-shuffle"]["prediction_records"].pop()
    reports["c100"]["evaluations"]["standard"]["prediction_records"][0]["sample_id"] = "wrong-order"

    contract = validate_contract(
        reports,
        report_paths=_report_paths(tmp_path),
        selection_report=_selection_report(),
    )

    assert contract["passed"] is False
    assert contract["checks"]["every_evaluation_has_ordered_val64_predictions"] is False


def test_paired_prediction_change_uses_xy_logits_and_selected_views():
    baseline = _records(64)
    changed = deepcopy(baseline)
    changed[0]["pred_xy"][0][0] = [11, 10]
    changed[1]["visibility_logits"][1] = [5.0, 0.0, 1.0, -1.0]

    result = paired_prediction_change(baseline, changed)

    assert result["sample_any_pred_xy_changed_fraction"] == pytest.approx(1 / 64)
    assert result["sample_any_visibility_logit_changed_fraction"] == pytest.approx(1 / 64)
    assert result["sample_any_selected_view_changed_fraction"] == pytest.approx(1 / 64)
    assert result["sample_any_prediction_changed_fraction"] == pytest.approx(2 / 64)


def test_main_writes_json_and_csv(tmp_path, monkeypatch):
    reports = _reports()
    paths = _report_paths(tmp_path)
    for role, path in paths.items():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(reports[role], allow_nan=True), encoding="utf-8")
    selection_path = tmp_path / "selection_report.json"
    selection_path.write_text(json.dumps(_selection_report()), encoding="utf-8")
    output_dir = tmp_path / "summary"
    argv = ["summarize"]
    for role in REPORT_ROLES:
        argv.extend([f"--{role}-report", str(paths[role])])
    argv.extend(["--selection-report", str(selection_path), "--output-dir", str(output_dir)])
    monkeypatch.setattr(sys, "argv", argv)

    assert main() == 0
    payload = json.loads((output_dir / "task35b_debiased_model_summary.json").read_text(encoding="utf-8"))
    assert payload["contract"]["passed"] is True
    assert (output_dir / "task35b_debiased_model_summary.csv").is_file()

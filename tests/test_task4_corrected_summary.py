from __future__ import annotations

import hashlib
import json
import sys
from copy import deepcopy

import pytest

from scripts.tools.summarize_task4_corrected_pilot import (
    BRANCH_B,
    BRANCH_C,
    LOSS_REDUCTION,
    POOL_MODE,
    STREAM_ALGORITHM,
    _hash_strings,
    build_summary,
    engineering_gate,
    extract_trajectories,
    main,
    scientific_gate,
    validate_contract,
    validate_planned_stream,
)


def _heatmap_selection(prefix: str, count: int):
    identities = [f"{prefix}/{index}" for index in range(count)]
    return {
        "sample_count": count,
        "sample_identity_sha256": hashlib.sha256(
            "\n".join(identities).encode("utf-8")
        ).hexdigest(),
        "sample_identities": identities,
        "scenes": [f"{prefix}_scene"],
    }


def _sft_selection(prefix: str, count: int, *, pixel: int, stop: int):
    identities = [f"{prefix}/{index}" for index in range(count)]
    return {
        "sample_count": count,
        "unique_physical_sample_count": count,
        "duplicate_physical_sample_count": 0,
        "sample_identity_sha256": _hash_strings(identities),
        "sample_identities": identities,
        "scenes": ["scene_00"],
        "category_counts": {"pixel": pixel, "stop": stop},
    }


def _dataset_contract():
    scenes = [f"scene_{index:02d}" for index in range(61)]
    counts = {
        scene: 9 if index < 12 else 8
        for index, scene in enumerate(scenes)
    }
    identities = [
        f"train/{scene}/clip_{clip:03d}"
        for scene in scenes
        for clip in range(counts[scene])
    ]
    assert len(identities) == 500
    return {
        "clip_count": 500,
        "scene_count": 61,
        "scenes": scenes,
        "per_scene_clip_counts": counts,
        "clip_identities": identities,
        "clip_identity_sha256": _hash_strings(identities),
        "balanced_view_manifest": {"selected_clip_identity_sha256": "balanced"},
    }


def _planned_stream():
    batches = []
    for step in range(100):
        indices = list(range(step * 4, step * 4 + 4))
        identities = [f"planned/{index}" for index in indices]
        batches.append(
            {
                "epoch": 0,
                "start_position": step * 4,
                "dataset_indices": indices,
                "sample_identities": identities,
                "category_counts": {"pixel": 3, "stop": 1},
            }
        )
    flat_indices = [index for batch in batches for index in batch["dataset_indices"]]
    flat_identities = [value for batch in batches for value in batch["sample_identities"]]
    return {
        "algorithm": STREAM_ALGORITHM,
        "seed": 4046,
        "batch_size": 4,
        "no_replacement_within_epoch": True,
        "candidate_count": 7995,
        "candidate_dataset_index_sha256": _hash_strings(str(index) for index in range(7995)),
        "planned_steps": 100,
        "planned_sample_count": 400,
        "planned_epoch_count": 1,
        "planned_dataset_index_sha256": _hash_strings(str(index) for index in flat_indices),
        "planned_sample_identity_sha256": _hash_strings(flat_identities),
        "planned_category_counts": {"pixel": 300, "stop": 100},
        "planned_batches": batches,
    }


def _heatmap_metrics(*, median: float, pck8: float, joint_pck8: float):
    return {
        "loss": 1.0,
        "visibility_auroc": 0.8,
        "visibility_auprc": 0.7,
        "visibility_f1": 0.6,
        "visibility_precision": 0.7,
        "visibility_recall": 0.6,
        "visible_view_accuracy": 0.75,
        "median_pixel_error": median,
        "median_u_error": median,
        "pck4": pck8 - 0.2,
        "pck8": pck8,
        "joint_median_pixel_error": median + 1.0,
        "joint_pck4": joint_pck8 - 0.2,
        "joint_pck8": joint_pck8,
        "samples": 64,
        "visible_history_count": 84,
        "visible_view_count": 84,
    }


def _heatmap_evaluations(*, median: float, pck8: float, joint_pck8: float):
    standard = _heatmap_metrics(median=median, pck8=pck8, joint_pck8=joint_pck8)
    return {
        "standard": standard,
        "blank-images": _heatmap_metrics(median=40.0, pck8=0.0, joint_pck8=0.0),
        "history-shuffle": _heatmap_metrics(
            median=median + 3.0,
            pck8=max(0.0, pck8 - 0.2),
            joint_pck8=max(0.0, joint_pck8 - 0.2),
        ),
        "current-shuffle": _heatmap_metrics(
            median=median + 2.0,
            pck8=max(0.0, pck8 - 0.1),
            joint_pck8=max(0.0, joint_pck8 - 0.1),
        ),
    }


def _generation(value: float = 1.0):
    return {
        "samples": 64,
        "requested_samples": 64,
        "attempted_samples": 64,
        "errors": 0,
        "skipped_no_target": 0,
        "complete_coverage": True,
        "format_valid": value,
        "action_valid": value,
        "category_match": value,
        "coord_hit": value,
        "view_hit": value,
        "stop_hit": value,
        "turn_hit": 0.0,
        "counts": {"total": 64.0},
    }


def _ce(loss: float):
    return {
        "loss": loss,
        "perplexity": 1.0 + loss,
        "samples": 64,
        "label_tokens": 768,
    }


def _lora_drift():
    return {
        "frozen_late_layers_unchanged": True,
        "layers": {
            str(layer): {
                "changed_tensors": 8 if layer <= 20 else 0,
                "parameter_delta_norm": 1.0 if layer <= 20 else 0.0,
            }
            for layer in range(28)
        },
    }


def _common_contract():
    scenes = _dataset_contract()["scenes"]
    rehearsal = scenes[:-7]
    holdout = scenes[-7:]
    sft_rehearsal = _sft_selection("candidate", 7995, pixel=6000, stop=1995)
    sft_rehearsal["scenes"] = rehearsal
    sft_rehearsal["pool_mode"] = POOL_MODE
    sft_rehearsal["full_candidate_count_before_optional_cap"] = 7995
    sft_holdout = _sft_selection("holdout", 64, pixel=48, stop=16)
    sft_holdout["scenes"] = holdout
    return {
        "initial_head_hash": "head0",
        "fresh_initial_head_hash": "head0",
        "starting_head_hash": "head0",
        "final_head_hash": "head0",
        "initial_lora_hash": "lora0",
        "final_lora_hash": "lora0",
        "all_lora_tensors": 224,
        "trainable_lora_layers": [],
        "max_trainable_lora_layer": 20,
        "frozen_late_layers_unchanged": True,
        "heatmap_train": _heatmap_selection("hm_train", 128),
        "heatmap_val": _heatmap_selection("hm_val", 64),
        "sft_dataset": _dataset_contract(),
        "sft_scene_partition": {
            "requested_holdout_scene_count": 7,
            "rehearsal_scenes": rehearsal,
            "holdout_scenes": holdout,
        },
        "sft_rehearsal": sft_rehearsal,
        "sft_retention": sft_holdout,
    }


def _eval_report(
    checkpoint: str,
    *,
    ce_loss: float,
    generation: float,
    median: float,
    pck8: float,
    joint_pck8: float,
):
    return {
        "task": "task4_joint_pilot",
        "mode": "head-only",
        "train_steps": 0,
        "checkpoint": checkpoint,
        "load": {"matched_lora_tensors": 224},
        "contract": _common_contract(),
        "lora_drift": _lora_drift(),
        "heatmap_evaluations": _heatmap_evaluations(
            median=median,
            pck8=pck8,
            joint_pck8=joint_pck8,
        ),
        "sft_retention": {
            "teacher_forced_before": _ce(ce_loss),
            "teacher_forced_after": _ce(ce_loss),
            "generation_before": _generation(generation),
            "generation_after": _generation(generation),
        },
    }


def _final_report(mode: str):
    is_joint = mode == BRANCH_C
    stream = _planned_stream()
    contract = _common_contract()
    contract.update(
        {
            "final_head_hash": f"{mode}-head100",
            "final_lora_hash": f"{mode}-lora100",
            "trainable_lora_layers": list(range(21)),
            "sft_rehearsal_stream": stream,
            "milestones": {
                "requested_steps": [0, 25, 50, 100],
                "effective_steps": [0, 25, 50, 100],
                "midpoint_evaluation_in_training_process": False,
            },
            "training_telemetry": {
                "record_count": 100,
                "expected_record_count": 100,
                "every_optimizer_step_recorded": True,
                "executed_sft_steps": 100 if is_joint else 0,
                "expected_executed_sft_steps": 100 if is_joint else 0,
                "total_sft_label_tokens": 800 if is_joint else 0,
            },
        }
    )
    train_log = []
    for step, planned in enumerate(stream["planned_batches"], start=1):
        train_log.append(
            {
                "step": step,
                "lm_label_tokens": 8 if is_joint else 0,
                "lm_sample_label_tokens": [2, 2, 2, 2] if is_joint else [],
                "sft_rehearsal_batch": {
                    "executed": is_joint,
                    **deepcopy(planned),
                },
            }
        )
    if is_joint:
        ce_after = 0.108
        generation_after = 0.99
        median, pck8, joint_pck8 = 5.2, 0.78, 0.74
    else:
        ce_after = 0.30
        generation_after = 0.90
        median, pck8, joint_pck8 = 5.0, 0.80, 0.75
    optimization = {
        "head_learning_rate": 1e-4,
        "lora_learning_rate": 1e-4,
        "rehearsal_weight": 1.0,
        "weight_decay": 1e-2,
        "grad_clip": 1.0,
        "lambda_coord": 0.0,
        "sft_batch_size": 4,
        "sft_pool_mode": POOL_MODE,
        "sft_stream_algorithm": STREAM_ALGORITHM,
        "sft_loss_reduction": LOSS_REDUCTION,
    }
    return {
        "task": "task4_joint_pilot",
        "mode": mode,
        "train_steps": 100,
        "checkpoint": f"/runs/{mode}/checkpoint_final.pth",
        "load": {"matched_lora_tensors": 224},
        "contract": contract,
        "optimization": optimization,
        "train_log": train_log,
        "lora_drift": _lora_drift(),
        "heatmap_evaluations": _heatmap_evaluations(
            median=median,
            pck8=pck8,
            joint_pck8=joint_pck8,
        ),
        "sft_retention": {
            "teacher_forced_before": _ce(0.10),
            "teacher_forced_after": _ce(ce_after),
            "generation_before": _generation(1.0),
            "generation_after": _generation(generation_after),
        },
    }


def _reports():
    return {
        "step0": _eval_report(
            "/runs/heatmap-lora/checkpoint_step_000000.pth",
            ce_loss=0.10,
            generation=1.0,
            median=10.0,
            pck8=0.50,
            joint_pck8=0.45,
        ),
        "b25": _eval_report(
            "/runs/heatmap-lora/checkpoint_step_000025.pth",
            ce_loss=0.15,
            generation=0.97,
            median=8.0,
            pck8=0.60,
            joint_pck8=0.55,
        ),
        "b50": _eval_report(
            "/runs/heatmap-lora/checkpoint_step_000050.pth",
            ce_loss=0.20,
            generation=0.94,
            median=6.5,
            pck8=0.70,
            joint_pck8=0.65,
        ),
        "c25": _eval_report(
            "/runs/joint-rehearsal/checkpoint_step_000025.pth",
            ce_loss=0.102,
            generation=0.995,
            median=8.2,
            pck8=0.59,
            joint_pck8=0.54,
        ),
        "c50": _eval_report(
            "/runs/joint-rehearsal/checkpoint_step_000050.pth",
            ce_loss=0.104,
            generation=0.995,
            median=6.7,
            pck8=0.69,
            joint_pck8=0.64,
        ),
        "b100": _final_report(BRANCH_B),
        "c100": _final_report(BRANCH_C),
    }


def _task35b(*, ready: bool = True, weaker: bool = True):
    return {
        "task": "task35b_debiased_data_diagnostic",
        "selection_ready_for_diagnostic": ready,
        "empirical_prior_strength": {
            "comparison": {
                "available": True,
                "shortcut_reduction": {
                    "empirical_prior_weaker_on_all_localization_checks": weaker,
                },
            },
        },
    }


def test_corrected_contract_validates_full_stream_and_all_reports():
    contract = validate_contract(_reports())
    assert contract["passed"]
    assert all(contract["checks"].values())
    stream = contract["planned_stream"]
    assert stream["passed"]
    assert stream["recomputed"]["unique_planned_dataset_indices"] == 400
    assert all(value["complete"] for value in contract["generation_coverage"].values())


def test_stream_validation_recomputes_hashes_and_rejects_tampering():
    stream = _planned_stream()
    assert validate_planned_stream(stream)["passed"]
    stream["planned_batches"][10]["sample_identities"][0] = "tampered"
    result = validate_planned_stream(stream)
    assert not result["passed"]
    assert not result["checks"]["planned_identity_hash_recomputed"]


def test_contract_rejects_incomplete_candidate_or_generation_coverage():
    reports = _reports()
    reports["c100"]["contract"]["sft_rehearsal_stream"]["candidate_count"] = 7994
    assert not validate_contract(reports)["passed"]

    reports = _reports()
    reports["c50"]["sft_retention"]["generation_after"]["errors"] = 1
    contract = validate_contract(reports)
    assert not contract["passed"]
    assert not contract["checks"]["all_generation_coverage_complete"]


def test_trajectory_contains_standard_interventions_and_sft_metrics_at_all_steps():
    trajectories = extract_trajectories(_reports())
    assert set(trajectories) == {BRANCH_B, BRANCH_C}
    assert set(trajectories[BRANCH_B]) == {"0", "25", "50", "100"}
    assert trajectories[BRANCH_B]["25"]["heatmap"]["standard"]["pck8"] == 0.60
    assert "history-shuffle" in trajectories[BRANCH_C]["50"]["heatmap"]
    assert trajectories[BRANCH_C]["100"]["sft"]["teacher_forced"]["loss"] == 0.108
    assert trajectories[BRANCH_C]["100"]["sft"]["generation"]["category_match"] == 0.99


def test_engineering_gate_applies_heatmap_ce_and_generation_thresholds():
    reports = _reports()
    contract = validate_contract(reports)
    gate = engineering_gate(reports, contract)
    assert gate["passed"]
    assert gate["metrics"]["c_retention_ce"]["relative_increase"] == pytest.approx(0.08)
    assert gate["metrics"]["c_generation"]["critical_drop"] == pytest.approx(-0.01)

    reports["c100"]["sft_retention"]["teacher_forced_after"]["loss"] = 0.12
    failed = engineering_gate(reports, contract)
    assert not failed["passed"]
    assert not failed["checks"]["c_retention_ce_within_ten_percent"]


def test_scientific_gate_is_pending_without_report_and_strict_with_report():
    pending = scientific_gate(None)
    assert pending["status"] == "pending"
    assert pending["passed"] is None

    passed = scientific_gate(_task35b())
    assert passed["status"] == "passed"
    assert passed["passed"] is True

    failed = scientific_gate(_task35b(weaker=False))
    assert failed["status"] == "failed"
    assert failed["passed"] is False


def test_build_summary_keeps_engineering_and_scientific_verdicts_separate():
    summary = build_summary(_reports(), task35b=None)
    assert summary["engineering_gate"]["passed"]
    assert summary["scientific_gate"]["status"] == "pending"
    assert not summary["verdict"]["advance_to_task5"]

    completed = build_summary(_reports(), task35b=_task35b())
    assert completed["verdict"]["advance_to_task5"]


def test_cli_writes_json_and_eight_row_trajectory_csv(tmp_path, monkeypatch):
    reports = _reports()
    paths = {}
    for name, report in reports.items():
        path = tmp_path / f"{name}.json"
        path.write_text(json.dumps(report), encoding="utf-8")
        paths[name] = path
    output_dir = tmp_path / "summary"
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "summarize_task4_corrected_pilot.py",
            "--heatmap-lora-final-report", str(paths["b100"]),
            "--joint-final-report", str(paths["c100"]),
            "--shared-step0-report", str(paths["step0"]),
            "--heatmap-lora-step25-report", str(paths["b25"]),
            "--heatmap-lora-step50-report", str(paths["b50"]),
            "--joint-step25-report", str(paths["c25"]),
            "--joint-step50-report", str(paths["c50"]),
            "--output-dir", str(output_dir),
        ],
    )
    assert main() == 0
    payload = json.loads(
        (output_dir / "task4_corrected_summary.json").read_text(encoding="utf-8")
    )
    assert payload["engineering_gate"]["passed"]
    assert payload["scientific_gate"]["status"] == "pending"
    assert len(
        (output_dir / "task4_corrected_trajectory.csv")
        .read_text(encoding="utf-8")
        .splitlines()
    ) == 9

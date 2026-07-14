from __future__ import annotations

import copy
import hashlib
from typing import Any

import pytest
from scripts.tools.summarize_pose_free_visual_identity import (
    CELL_NAMES,
    CHECKPOINT_SCHEMA,
    EXPECTED_PAIR_MATCHED_CONTRACTS,
    HISTORY_QUERY_SOURCE,
    INTERVENTIONS,
    PEAK_RECONSTRUCTION_CONTRACT,
    REPORT_PROTOCOL,
    REPORT_SCHEMA,
    SCORE_RECONSTRUCTION_CONTRACT,
    SUMMARY_SCHEMA,
    TRAIN_PROTOCOL,
    VISIBILITY_RECONSTRUCTION_CONTRACT,
    record_counts,
    summarize,
)

WIDTH = 64


def _sha(label: str) -> str:
    return hashlib.sha256(label.encode()).hexdigest()


def _gt() -> tuple[list[list[float]], list[list[list[int]]]]:
    visibility = [[0.0] * 4 for _ in range(4)]
    xy = [[[0, 0] for _ in range(4)] for _ in range(4)]
    for slot in range(4):
        visibility[slot][slot] = 1.0
        xy[slot][slot] = [10 + slot, 12 + slot]
    return visibility, xy


def _record(sample_id: str, assignments: list[int], *, target_slot: int | None = None) -> dict[str, Any]:
    gt_visibility, gt_xy = _gt()
    visibility_logits: list[list[float]] = []
    pred_xy: list[list[list[int]]] = []
    score: list[list[float]] = []
    for assigned in assignments:
        row_visibility = [-6.0] * 4
        row_visibility[assigned] = 6.0
        visibility_logits.append(row_visibility)
        row_xy = [[0, 0] for _ in range(4)]
        row_xy[assigned] = list(gt_xy[assigned][assigned])
        pred_xy.append(row_xy)
        row_score = [-8.0] * 4
        row_score[assigned] = -0.1
        score.append(row_score)
    return {
        "sample_id": sample_id,
        "target_slot": target_slot,
        "visibility_logits": visibility_logits,
        "gt_visibility": gt_visibility,
        "pred_xy": pred_xy,
        "gt_xy": gt_xy,
        "target_score_matrix": score,
        "score_reconstruction": copy.deepcopy(SCORE_RECONSTRUCTION_CONTRACT),
        "visibility_reconstruction": copy.deepcopy(VISIBILITY_RECONSTRUCTION_CONTRACT),
        "peak_reconstruction": copy.deepcopy(PEAK_RECONSTRUCTION_CONTRACT),
        "global_pred_view_xy": [
            [assigned, pred_xy[row][assigned][0], pred_xy[row][assigned][1]] for row, assigned in enumerate(assignments)
        ],
    }


def _swap_records(sample_id: str, standard_assignments: list[int]) -> list[dict[str, Any]]:
    records = []
    for target_slot in range(4):
        assignments = list(standard_assignments)
        assignments[target_slot] = (target_slot + 1) % 4
        records.append(_record(sample_id, assignments, target_slot=target_slot))
    return records


def _metric(records: list[dict[str, Any]], *, targeted: bool = False) -> dict[str, Any]:
    totals: dict[str, list[float]] = {}
    for record in records:
        slots = [record["target_slot"]] if targeted else None
        counts = record_counts(record, width=WIDTH, slots=slots)
        for name, pair in counts.items():
            aggregate = totals.setdefault(name, [0.0, 0.0])
            aggregate[0] += pair[0]
            aggregate[1] += pair[1]

    def value(name: str) -> float:
        return totals[name][0] / totals[name][1]

    return {
        "anchor_identity": {"accuracy": value("peak_nearest_identity_accuracy")},
        "visible_view_accuracy": value("visible_view_accuracy"),
        "pck8": value("conditional_pck8"),
        "joint_pck8": value("true_joint_pck8"),
        "global_map_joint_pck8": value("global_map_joint_pck8"),
    }


def _routing(source_samples: int) -> dict[str, Any]:
    return {
        "contract": "replace history i; compare output i against all output j!=i on the same current",
        "targeted": {
            "comparisons": source_samples * 4,
            "mean_heatmap_l1": 0.2,
            "mean_visibility_l1": 0.3,
            "mean_peak_displacement": 20.0,
        },
        "untargeted": {
            "comparisons": source_samples * 12,
            "mean_heatmap_l1": 0.0,
            "mean_visibility_l1": 0.0,
            "mean_peak_displacement": 0.0,
        },
        "targeted_to_untargeted_heatmap_l1_ratio": None,
    }


def _prediction_records(cell: str, source_samples: int = 40) -> dict[str, list[dict[str, Any]]]:
    standard_assignments = {
        "identity-trained": [0, 1, 2, 3],
        "heatmap-control-trained": [0, 1, 0, 0],
        "warmup-original": [0, 0, 0, 0],
    }[cell]
    changed_assignments = [1, 2, 3, 0] if cell == "identity-trained" else standard_assignments
    output: dict[str, list[dict[str, Any]]] = {intervention: [] for intervention in INTERVENTIONS}
    for position in range(source_samples):
        sample_id = f"scene{position:02d}/clip{position:04d}"
        output["standard"].append(_record(sample_id, standard_assignments))
        output["history-shuffle"].append(_record(sample_id, changed_assignments))
        output["current-shuffle"].append(_record(sample_id, changed_assignments))
        output["blank-images"].append(_record(sample_id, [0, 0, 0, 0]))
        output["single-anchor-swap"].extend(_swap_records(sample_id, standard_assignments))
    return output


def _gates(source_samples: int = 40) -> dict[str, Any]:
    history = {
        "passed": True,
        "bitwise_exact": True,
        "samples": source_samples,
        "tensor_comparisons": source_samples * 2,
        "maximum_abs_difference": 0.0,
    }
    current = {
        "passed": True,
        "paired_source_samples": source_samples,
        "sample_order_exact": True,
    }
    swap = {
        "passed": True,
        "bitwise_exact": True,
        "source_samples": source_samples,
        "swap_pairs": source_samples * 4,
        "untargeted_output_slots": source_samples * 12,
        "tensor_comparisons": source_samples * 24,
        "maximum_abs_difference": 0.0,
    }
    blank_input = {"passed": True, "bitwise_exact": True, "samples": source_samples}
    blank_output = {
        "passed": True,
        "bitwise_exact": True,
        "samples": source_samples,
        "maximum_abs_difference": 0.0,
    }
    return {
        "standard": {
            "passed": True,
            "source_samples": source_samples,
            "unique_sample_ids": True,
        },
        "history-shuffle": history,
        "current-shuffle": current,
        "single-anchor-swap": swap,
        "blank-images": {"passed": True, "input": blank_input, "output": blank_output},
    }


def _evaluations(records: dict[str, list[dict[str, Any]]], gates: dict[str, Any]) -> dict[str, Any]:
    source_samples = len(records["standard"])
    evaluations: dict[str, Any] = {}
    for intervention in INTERVENTIONS:
        metrics = _metric(records[intervention])
        metrics["samples"] = len(records[intervention])
        evaluations[intervention] = metrics
    evaluations["history-shuffle"]["permutation_equivariance_gate"] = gates["history-shuffle"]
    evaluations["current-shuffle"]["paired_schedule_gate"] = gates["current-shuffle"]
    evaluations["single-anchor-swap"]["targeted_slot_metrics"] = _metric(records["single-anchor-swap"], targeted=True)
    evaluations["single-anchor-swap"]["untargeted_invariance_gate"] = gates["single-anchor-swap"]
    evaluations["single-anchor-swap"]["paired_output_change_vs_standard"] = _routing(source_samples)
    evaluations["blank-images"]["blank_input_identity_gate"] = gates["blank-images"]["input"]
    evaluations["blank-images"]["blank_output_identity_gate"] = gates["blank-images"]["output"]
    return evaluations


def _shared_contracts(
    source_samples: int = 40,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], dict[str, Any]]:
    stage1 = {
        "path": "/pinned/stage1.pth",
        "file_sha256": _sha("stage1-file"),
        "matched_lora_tensors": 224,
        "loaded_lora_sha256": _sha("stage1-lora"),
    }
    manifest = {
        "manifest_sha256": _sha("manifest-semantic"),
        "file_sha256": _sha("manifest-file"),
        "source_inventory_sha256": _sha("inventory"),
        "max_clip_id": 2000,
        "source_inventory_clips": 2000,
        "num_history": 4,
        "train_identity_sha256": _sha("train-ids"),
        "val_identity_sha256": _sha("val-ids"),
        "train_samples": 128,
        "val_samples": source_samples,
        "scene_disjoint": True,
        "minimum_target_separation_pixels": 12.0,
        "identity_targets_per_sample": 4,
    }
    config = {
        "decoder_mode": "pose_free_matcher",
        "trajectory_enabled": False,
        "vit_layer_indices": [],
        "llm_layer_indices": [20],
        "model_pose_input": None,
        "isolated_pair_chains": True,
        "histories_per_qwen_chain": 1,
        "qwen_forward_batch_size": 1,
        "qwen_forwards_per_sample": 4,
        "protocol": TRAIN_PROTOCOL,
        "history_query_source": HISTORY_QUERY_SOURCE,
        "history_query_layer": 20,
        "history_visual_views_per_query": 4,
        "history_visual_view_reduction": "equal_weight_mean",
        "raw_heatmap_logits_required": True,
    }
    runtime = {
        "decoder_mode": "pose_free_matcher",
        "trajectory_enabled": False,
        "vit_hooks": [],
        "llm_hooks": [20],
        "matcher_uses_relative_pose": False,
        "head_trainable_parameters": 100,
        "isolated_pair_chains": True,
        "histories_per_qwen_chain": 1,
        "history_anchor_number_per_chain": 1,
        "qwen_forward_batch_size": 1,
        "qwen_forwards_per_sample": 4,
        "protocol": TRAIN_PROTOCOL,
        "history_query_source": HISTORY_QUERY_SOURCE,
        "history_query_layer": 20,
        "history_visual_views_per_query": 4,
        "raw_heatmap_logits_opt_in": "return_heatmap_logits=True",
    }
    return stage1, manifest, config, runtime


def _pair_gate(identity_file: str, control_file: str) -> dict[str, Any]:
    return {
        "passed": True,
        "identity_checkpoint": {"path": "/pilot/identity.pth", "file_sha256": identity_file},
        "control_checkpoint": {"path": "/pilot/control.pth", "file_sha256": control_file},
        "matched_contracts": sorted(EXPECTED_PAIR_MATCHED_CONTRACTS),
        "only_registered_difference": {
            "identity_weight": [2.0, 0.0],
        },
    }


def _reports(source_samples: int = 40) -> dict[str, dict[str, Any]]:
    stage1, manifest, config, runtime = _shared_contracts(source_samples)
    warmup_file = _sha("warmup-file")
    warmup_head = _sha("warmup-head")
    warmup_lora = stage1["loaded_lora_sha256"]
    identity_file = _sha("identity-file")
    control_file = _sha("control-file")
    pair = _pair_gate(identity_file, control_file)
    warmup_contract = {
        "schema": CHECKPOINT_SCHEMA,
        "protocol": TRAIN_PROTOCOL,
        "file_sha256": warmup_file,
        "head_state_sha256": warmup_head,
        "lora_state_sha256": warmup_lora,
        "step": 128,
        "training_sample_schedule_sha256": _sha("warmup-schedule"),
        "optimization_contract": {"train_steps": 128, "optimizer": "AdamW"},
    }
    reports: dict[str, dict[str, Any]] = {}
    for index, cell in enumerate(CELL_NAMES):
        records = _prediction_records(cell, source_samples)
        gates = _gates(source_samples)
        mode = {
            "warmup-original": "head-warmup",
            "identity-trained": "lora-identity",
            "heatmap-control-trained": "lora-heatmap-control",
        }[cell]
        if cell == "warmup-original":
            lora_source = {
                "source": "stage1-s2",
                "path": stage1["path"],
                "file_sha256": stage1["file_sha256"],
                "lora_state_sha256": stage1["loaded_lora_sha256"],
            }
            selected_file = warmup_file
            pair_contract = None
            identity_weight, panorama_weight = 0.0, 0.0
        elif cell == "identity-trained":
            lora_source = {
                "source": "lora-identity",
                "path": "/pilot/identity.pth",
                "file_sha256": identity_file,
                "lora_state_sha256": _sha("identity-lora"),
            }
            selected_file = identity_file
            pair_contract = copy.deepcopy(pair)
            identity_weight, panorama_weight = 2.0, 1.0
        else:
            lora_source = {
                "source": "lora-heatmap-control",
                "path": "/pilot/control.pth",
                "file_sha256": control_file,
                "lora_state_sha256": _sha("control-lora"),
            }
            selected_file = control_file
            pair_contract = copy.deepcopy(pair)
            identity_weight, panorama_weight = 0.0, 1.0
        head_source = {
            "source": "shared-head-warmup",
            "path": "/pilot/warmup.pth",
            "file_sha256": warmup_file,
            "head_state_sha256": warmup_head,
        }
        reports[cell] = {
            "schema": REPORT_SCHEMA,
            "protocol": REPORT_PROTOCOL,
            "phase": "eval",
            "cell": cell,
            "evaluation_scope": {
                "selection_split": "val",
                "standard_only": False,
                "source_samples": source_samples,
            },
            "evaluation_pid": 200 + index,
            "fresh_process_contract": {
                "training_pid": 100 + index,
                "evaluation_pid": 200 + index,
                "fresh_stage1_loaded_before_cell_state": True,
            },
            "stage1_s2_contract": copy.deepcopy(stage1),
            "manifest_contract": copy.deepcopy(manifest),
            "pose_free_config_contract": copy.deepcopy(config),
            "runtime_contract": copy.deepcopy(runtime),
            "selected_cell_contract": {
                "cell": cell,
                "expected_train_mode": mode,
                "selected_checkpoint_path": lora_source["path"] if cell != "warmup-original" else "/pilot/warmup.pth",
                "selected_checkpoint_file_sha256": selected_file,
                "training_pid": 100 + index,
                "step": 256 if cell != "warmup-original" else 128,
                "fresh_stage1_lora_loaded_before_cell_state": True,
                "fresh_stage1_lora_sha256": stage1["loaded_lora_sha256"],
                "shared_warmup_contract": copy.deepcopy(warmup_contract),
                "active_head_sha256": warmup_head,
                "active_lora_sha256": lora_source["lora_state_sha256"],
                "loss_contract": {
                    "base_weight": 1.0,
                    "identity_weight": identity_weight,
                    "panorama_weight": panorama_weight,
                    "panorama_objective": "global_raw_heatmap_pixel_ce",
                    "view_readout": "raw_heatmap_spatial_logsumexp_marginal",
                    "control_differs_only_by_identity_term": True,
                },
                "identity_control_pair_gate": pair_contract,
            },
            "checkpoint_sources": {"head": head_source, "lora": lora_source},
            "state_and_input_contract": {
                "explicit_pose_inputs_removed": True,
                "history_query_source": HISTORY_QUERY_SOURCE,
                "history_query_layer": 20,
                "history_visual_views_per_query": 4,
                "shared_head_across_cells": True,
                "qwen_forward_batch_size": 1,
                "qwen_forwards_per_sample": 4,
                "view_metric_source": "raw_heatmap_spatial_logsumexp_marginal",
                "learned_visibility_readout_used_for_view_metric": False,
            },
            "interventions": list(INTERVENTIONS),
            "intervention_gates": gates,
            "evaluations": _evaluations(records, gates),
            "prediction_records": records,
        }
    return reports


def test_summary_is_source_paired_and_authorizes_only_when_all_gates_pass():
    summary = summarize(_reports(), bootstrap_samples=250, seed=17, heatmap_width=WIDTH)

    assert summary["schema"] == SUMMARY_SCHEMA
    assert summary["contract_validation"]["passed"] is True
    assert summary["contract_validation"]["target_score_reconstruction_verified"] is True
    assert summary["bootstrap_contract"] == {
        "method": "paired_source_sample_percentile_bootstrap",
        "resampling_unit": "source_sample",
        "source_samples": 40,
        "replicates": 250,
        "seed": 17,
        "paired_across_cells_and_interventions": True,
        "four_history_outputs_remain_grouped": True,
        "single_anchor_swap_replicas_remain_grouped": True,
    }
    assert summary["cells"]["identity-trained"]["standard"]["score_nearest_identity_accuracy"]["estimate"] == 1.0
    assert summary["cells"]["identity-trained"]["standard"]["peak_nearest_identity_accuracy"]["estimate"] == 1.0
    assert (
        summary["causal_comparisons"]["identity-minus-control"]["metrics"]["score_nearest_identity_accuracy"][
            "estimate"
        ]
        == 0.5
    )
    assert (
        summary["causal_effects"]["identity-trained"]["standard-minus-history-shuffle"]["metrics"][
            "score_nearest_identity_accuracy"
        ]["estimate"]
        == 1.0
    )
    assert summary["decision_gate"]["overall_passed"] is True
    assert summary["decision_gate"]["overall_stage2_stage3_authorized"] is True
    assert summary["overall_stage2_stage3_authorized"] is True


def test_summary_derives_and_accepts_128_source_validation_contract():
    summary = summarize(_reports(128), bootstrap_samples=20, seed=19, heatmap_width=WIDTH)

    assert summary["contract_validation"]["source_samples"] == 128
    assert summary["bootstrap_contract"]["source_samples"] == 128
    assert summary["cells"]["identity-trained"]["standard"]["score_nearest_identity_accuracy"]["denominator"] == 512.0
    assert summary["overall_stage2_stage3_authorized"] is True


def test_summary_rejects_manifest_or_evaluation_scope_count_disagreement():
    reports = _reports(128)
    reports["heatmap-control-trained"]["manifest_contract"]["val_samples"] = 127
    reports["heatmap-control-trained"]["evaluation_scope"]["source_samples"] = 127
    with pytest.raises(RuntimeError, match="do not agree on manifest val_samples"):
        summarize(reports, bootstrap_samples=10, heatmap_width=WIDTH)

    reports = _reports(128)
    reports["identity-trained"]["evaluation_scope"]["standard_only"] = True
    with pytest.raises(RuntimeError, match="evaluation scope is not the complete manifest validation split"):
        summarize(reports, bootstrap_samples=10, heatmap_width=WIDTH)


def test_gate_fails_when_score_identity_does_not_beat_control():
    reports = _reports()
    for record in reports["identity-trained"]["prediction_records"]["standard"]:
        record["target_score_matrix"] = copy.deepcopy(_record("unused", [0, 0, 0, 0])["target_score_matrix"])

    summary = summarize(reports, bootstrap_samples=50, seed=3, heatmap_width=WIDTH)

    checks = summary["decision_gate"]["checks"]
    assert checks["identity_score_nearest_accuracy_at_least_0_45"]["passed"] is False
    assert checks["identity_score_nearest_gain_vs_control_at_least_0_10"]["passed"] is False
    assert summary["overall_stage2_stage3_authorized"] is False
    assert summary["decision_gate"]["failure_action"] is not None


def test_summary_rejects_record_schedule_shorter_than_manifest_contract():
    reports = _reports()
    for report in reports.values():
        for intervention in INTERVENTIONS:
            if intervention == "single-anchor-swap":
                del report["prediction_records"][intervention][-4:]
            else:
                report["prediction_records"][intervention].pop()

    with pytest.raises(RuntimeError, match="exactly 40 source IDs"):
        summarize(reports, bootstrap_samples=10, heatmap_width=WIDTH)


def test_summary_rejects_missing_score_matrix_or_reconstruction_drift():
    reports = _reports()
    del reports["identity-trained"]["prediction_records"]["standard"][0]["target_score_matrix"]
    with pytest.raises(RuntimeError, match="target_score_matrix must be"):
        summarize(reports, bootstrap_samples=10, heatmap_width=WIDTH)

    reports = _reports()
    reports["identity-trained"]["prediction_records"]["standard"][0]["score_reconstruction"]["raw_logits_opt_in"] = (
        "missing"
    )
    with pytest.raises(RuntimeError, match="score reconstruction contract mismatch"):
        summarize(reports, bootstrap_samples=10, heatmap_width=WIDTH)


def test_summary_rejects_non_shared_head_or_noncausal_checkpoint_pair():
    reports = _reports()
    reports["heatmap-control-trained"]["checkpoint_sources"]["head"]["file_sha256"] = _sha("different-head-file")
    with pytest.raises(RuntimeError, match="share one exact warmup head"):
        summarize(reports, bootstrap_samples=10, heatmap_width=WIDTH)

    reports = _reports()
    reports["heatmap-control-trained"]["selected_cell_contract"]["identity_control_pair_gate"][
        "matched_contracts"
    ].remove("training_sample_schedule_sha256")
    with pytest.raises(RuntimeError, match="disagree on causal pair"):
        summarize(reports, bootstrap_samples=10, heatmap_width=WIDTH)


def test_summary_rejects_failed_bitwise_intervention_gate():
    reports = _reports()
    gate = reports["identity-trained"]["intervention_gates"]["single-anchor-swap"]
    gate["bitwise_exact"] = False
    reports["identity-trained"]["evaluations"]["single-anchor-swap"]["untargeted_invariance_gate"] = gate

    with pytest.raises(RuntimeError, match="single-swap locality gate is not bitwise exact"):
        summarize(reports, bootstrap_samples=10, heatmap_width=WIDTH)


def test_summary_rejects_compact_metric_disagreeing_with_report():
    reports = _reports()
    reports["identity-trained"]["evaluations"]["standard"]["joint_pck8"] = 0.5

    with pytest.raises(RuntimeError, match="compact/reported true_joint_pck8 mismatch"):
        summarize(reports, bootstrap_samples=10, heatmap_width=WIDTH)


def test_summary_rejects_legacy_text_anchor_or_b4_runtime():
    reports = _reports()
    for report in reports.values():
        report["runtime_contract"]["history_query_source"] = "text_anchor"
        report["runtime_contract"]["qwen_forward_batch_size"] = 4

    with pytest.raises(RuntimeError, match="wrong query source"):
        summarize(reports, bootstrap_samples=10, heatmap_width=WIDTH)

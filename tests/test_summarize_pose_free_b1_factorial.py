import copy

import pytest
from scripts.tools.summarize_pose_free_b1_factorial import (
    INTERVENTIONS,
    METRICS,
    REPORT_SCHEMA,
    SUMMARY_SCHEMA,
    build_contributions,
    summarize,
)

SOURCE_IDS = [f"scene/sample-{index}" for index in range(4)]
STAGE1_LORA = {
    "path": "/checkpoints/stage1.pth",
    "file_sha256": "1" * 64,
    "branch": "stage1-s2",
    "head_state_sha256": None,
    "lora_state_sha256": "2" * 64,
}
HEAD_ONLY = {
    "path": "/pilots/head-only.pth",
    "file_sha256": "3" * 64,
    "branch": "head-only",
    "head_state_sha256": "4" * 64,
    "lora_state_sha256": "2" * 64,
}
JOINT = {
    "path": "/pilots/joint.pth",
    "file_sha256": "5" * 64,
    "branch": "heatmap-lora",
    "head_state_sha256": "6" * 64,
    "lora_state_sha256": "7" * 64,
}


def _record(sample_id, mapping, target_slot=None):
    gt_visibility = [[1.0, 0.0, 0.0, 0.0] for _ in range(4)]
    gt_xy = [[[0, 0] for _ in range(4)] for _ in range(4)]
    pred_xy = [[[0, 0] for _ in range(4)] for _ in range(4)]
    target_x = [6, 22, 38, 54]
    for slot in range(4):
        gt_xy[slot][0] = [target_x[slot], 10]
        for view in range(4):
            pred_xy[slot][view] = [target_x[mapping[slot]], 10]
    return {
        "sample_id": sample_id,
        "target_slot": target_slot,
        "visibility_logits": [[9.0, 0.0, 0.0, 0.0] for _ in range(4)],
        "gt_visibility": gt_visibility,
        "pred_xy": pred_xy,
        "gt_xy": gt_xy,
    }


def _records(mapping):
    standard = [_record(sample_id, mapping) for sample_id in SOURCE_IDS]
    reverse = list(reversed(mapping))
    history = [_record(sample_id, reverse) for sample_id in SOURCE_IDS]
    current = [_record(sample_id, [3, 3, 3, 3]) for sample_id in SOURCE_IDS]
    blank = [_record(sample_id, [0, 0, 0, 0]) for sample_id in SOURCE_IDS]
    swaps = []
    for sample_id in SOURCE_IDS:
        for target_slot in range(4):
            swapped = list(mapping)
            swapped[target_slot] = (target_slot + 1) % 4
            swaps.append(_record(sample_id, swapped, target_slot=target_slot))
    return {
        "standard": standard,
        "blank-images": blank,
        "history-shuffle": history,
        "current-shuffle": current,
        "single-anchor-swap": swaps,
    }


def _metrics(records, *, targeted=False):
    contributions = build_contributions(
        records,
        SOURCE_IDS,
        width=64,
        targeted_swap=targeted,
    )
    values = {metric: float(value[:, 0].sum() / value[:, 1].sum()) for metric, value in contributions.items()}
    return {
        "anchor_identity": {"accuracy": values["anchor_identity_accuracy"]},
        "visible_view_accuracy": values["visible_view_accuracy"],
        "pck4": values["conditional_pck4"],
        "pck8": values["conditional_pck8"],
        "joint_pck4": values["true_joint_pck4"],
        "joint_pck8": values["true_joint_pck8"],
    }


def _pilot_contract(branch, eval_lora, head, lora, override):
    base = HEAD_ONLY if branch == "head-only" else JOINT
    return {
        "path": base["path"],
        "file_sha256": base["file_sha256"],
        "branch": branch,
        "checkpoint_head_state_sha256": base["head_state_sha256"],
        "head_state_sha256": head["head_state_sha256"],
        "eval_lora": eval_lora,
        "active_lora_sha256": lora["lora_state_sha256"],
        "head_override": override,
        "head_source_checkpoint": copy.deepcopy(head),
        "lora_source_checkpoint": copy.deepcopy(lora),
    }


def _paired_output_change(*, targeted_heatmap_l1=1.0, targeted_peak_displacement=2.0):
    return {
        "targeted": {
            "comparisons": len(SOURCE_IDS) * 4,
            "mean_heatmap_l1": targeted_heatmap_l1,
            "mean_visibility_l1": 0.5,
            "mean_peak_displacement": targeted_peak_displacement,
        },
        "untargeted": {
            "comparisons": len(SOURCE_IDS) * 4 * 3,
            "mean_heatmap_l1": 0.0,
            "mean_visibility_l1": 0.0,
            "mean_peak_displacement": 0.0,
        },
        "targeted_to_untargeted_heatmap_l1_ratio": None,
        "contract": "replace history i; compare output i against all output j!=i on the same current",
    }


def _report(cell, mapping):
    roles = {
        "A": ("head-only", "off", HEAD_ONLY, STAGE1_LORA, False),
        "B": ("heatmap-lora", "trained", JOINT, JOINT, False),
        "C": ("heatmap-lora", "off", JOINT, STAGE1_LORA, False),
        "D": ("heatmap-lora", "trained", HEAD_ONLY, JOINT, True),
    }
    branch, eval_lora, head, lora, override = roles[cell]
    records = _records(mapping)
    evaluations = {
        intervention: _metrics(
            intervention_records,
            targeted=intervention == "single-anchor-swap",
        )
        for intervention, intervention_records in records.items()
    }
    for intervention, intervention_records in records.items():
        evaluations[intervention]["samples"] = len(intervention_records)
    evaluations["single-anchor-swap"]["targeted_slot_metrics"] = _metrics(records["single-anchor-swap"], targeted=True)
    evaluations["single-anchor-swap"]["paired_output_change_vs_standard"] = _paired_output_change()
    evaluations["blank-images"]["blank_input_identity_gate"] = {
        "passed": True,
        "bitwise_exact": True,
    }
    evaluations["blank-images"]["blank_output_identity_gate"] = {
        "passed": True,
        "bitwise_exact": True,
        "maximum_abs_difference": 0.0,
    }
    evaluations["history-shuffle"]["permutation_equivariance_gate"] = {
        "passed": True,
        "bitwise_exact": True,
        "maximum_abs_difference": 0.0,
    }
    return {
        "schema": REPORT_SCHEMA,
        "phase": "eval",
        "branch": branch,
        "eval_lora": eval_lora,
        "explicit_pose_inputs_removed": True,
        "interventions": list(INTERVENTIONS),
        "stage1_s2_contract": {
            "path": "/checkpoints/stage1.pth",
            "file_sha256": "1" * 64,
            "loaded_lora_sha256": "2" * 64,
            "matched_lora_tensors": 224,
        },
        "manifest_contract": {
            "manifest_sha256": "8" * 64,
            "val_identity_sha256": "9" * 64,
            "val_samples": len(SOURCE_IDS),
        },
        "pose_free_config_contract": {
            "isolated_pair_chains": True,
            "histories_per_qwen_chain": 1,
            "qwen_forward_batch_size": 1,
            "qwen_forwards_per_sample": 4,
            "model_pose_input": None,
        },
        "runtime_contract": {
            "isolated_pair_chains": True,
            "histories_per_qwen_chain": 1,
            "qwen_forward_batch_size": 1,
            "qwen_forwards_per_sample": 4,
            "matcher_uses_relative_pose": False,
        },
        "pilot_checkpoint": _pilot_contract(branch, eval_lora, head, lora, override),
        "checkpoint_sources": {
            "head": copy.deepcopy(head),
            "lora": copy.deepcopy(lora),
        },
        "evaluations": evaluations,
        "prediction_records": records,
    }


def _factorial_reports():
    return {
        "A": _report("A", [0, 0, 0, 0]),
        "B": _report("B", [0, 1, 2, 3]),
        "C": _report("C", [0, 1, 0, 1]),
        "D": _report("D", [0, 1, 2, 0]),
    }


def test_summary_pairs_by_source_sample_and_applies_predeclared_gates():
    summary = summarize(_factorial_reports(), bootstrap_samples=250, seed=17)

    assert summary["schema"] == SUMMARY_SCHEMA == "task36c_pose_free_b1_factorial_summary_v2"
    assert summary["contract_validation"]["passed"] is True
    assert summary["contract_validation"]["single_anchor_swap_untargeted_routing_exact"] is True
    assert summary["bootstrap_contract"]["resampling_unit"] == "source_sample"
    assert summary["bootstrap_contract"]["source_samples"] == 4
    assert summary["bootstrap_contract"]["single_anchor_swap_replicas_are_grouped_with_their_source_sample"]
    assert set(summary["factorial_comparisons"]) == {"B-A", "B-C", "D-A", "D-C"}
    assert set(summary["cells"]["B"]["standard"]) == set(METRICS)
    assert summary["cells"]["B"]["standard"]["anchor_identity_accuracy"]["estimate"] == 1.0
    assert summary["factorial_comparisons"]["B-A"]["metrics"]["anchor_identity_accuracy"]["estimate"] == 0.75
    assert summary["factorial_comparisons"]["D-A"]["metrics"]["anchor_identity_accuracy"]["estimate"] == 0.5
    assert (
        summary["causal_effects"]["B"]["standard_minus_history-shuffle"]["metrics"]["anchor_identity_accuracy"][
            "estimate"
        ]
        == 1.0
    )
    assert summary["cells"]["B"]["all_same_peak_fraction"]["estimate"] == 0.0
    assert summary["decision_gate"]["checks"]["B_minus_C_pure_LoRA_identity_gain"]["passed"]
    assert summary["decision_threshold_contract"]["predeclared_and_not_cli_tunable"]
    assert summary["decision_threshold_contract"]["conditional_pck8_minimum"] == 0.30
    assert summary["decision_gate"]["overall_passed"] is True
    assert summary["decision_gate"]["stage2_stage3_authorized_by_this_gate"] is True
    assert summary["decision_gate"]["failure_action"] is None


def test_summary_rejects_old_b4_runtime_contract():
    reports = _factorial_reports()
    for report in reports.values():
        report["runtime_contract"]["qwen_forward_batch_size"] = 4

    with pytest.raises(RuntimeError, match="Qwen forward batch is not B=1"):
        summarize(reports, bootstrap_samples=10)


def test_summary_rejects_unpaired_factorial_checkpoint_sources():
    reports = _factorial_reports()
    reports["D"]["checkpoint_sources"]["head"]["file_sha256"] = "a" * 64
    reports["D"]["pilot_checkpoint"]["head_source_checkpoint"]["file_sha256"] = "a" * 64

    with pytest.raises(RuntimeError, match="A/D do not use the same head-only head"):
        summarize(reports, bootstrap_samples=10)


def test_summary_rejects_compact_metric_that_disagrees_with_report():
    reports = _factorial_reports()
    reports["B"]["evaluations"]["standard"]["anchor_identity"]["accuracy"] = 0.25

    with pytest.raises(RuntimeError, match="compact/reported anchor_identity_accuracy mismatch"):
        summarize(reports, bootstrap_samples=10)


def test_summary_gate_fails_without_lora_retention_in_d():
    reports = _factorial_reports()
    reports["D"] = _report("D", [0, 0, 0, 0])

    summary = summarize(reports, bootstrap_samples=100, seed=3)

    check = summary["decision_gate"]["checks"]["D_minus_A_retains_LoRA_identity_gain"]
    assert check["passed"] is False
    assert summary["decision_gate"]["overall_passed"] is False
    assert summary["decision_gate"]["stage2_stage3_authorized_by_this_gate"] is False
    assert summary["decision_gate"]["failure_action"] is not None


def test_summary_gate_rejects_no_current_or_single_swap_causal_effect():
    reports = _factorial_reports()
    standard = reports["B"]["prediction_records"]["standard"]

    current_records = copy.deepcopy(standard)
    reports["B"]["prediction_records"]["current-shuffle"] = current_records
    current_evaluation = _metrics(current_records)
    current_evaluation["samples"] = len(current_records)
    reports["B"]["evaluations"]["current-shuffle"] = current_evaluation

    swap_records = []
    for record in standard:
        for target_slot in range(4):
            swapped = copy.deepcopy(record)
            swapped["target_slot"] = target_slot
            swap_records.append(swapped)
    reports["B"]["prediction_records"]["single-anchor-swap"] = swap_records
    swap_evaluation = _metrics(swap_records, targeted=True)
    swap_evaluation["samples"] = len(swap_records)
    swap_evaluation["targeted_slot_metrics"] = _metrics(swap_records, targeted=True)
    swap_evaluation["paired_output_change_vs_standard"] = _paired_output_change(
        targeted_heatmap_l1=0.0,
        targeted_peak_displacement=0.0,
    )
    reports["B"]["evaluations"]["single-anchor-swap"] = swap_evaluation

    summary = summarize(reports, bootstrap_samples=250, seed=17)
    checks = summary["decision_gate"]["checks"]

    assert checks["B_current_shuffle_true_joint_PCK8_drop_at_least_0_05"]["passed"] is False
    assert checks["B_targeted_single_swap_identity_drop_at_least_0_05"]["passed"] is False
    assert checks["B_targeted_single_swap_heatmap_L1_positive"]["passed"] is False
    assert checks["B_targeted_single_swap_peak_displacement_positive"]["passed"] is False
    assert summary["decision_gate"]["overall_passed"] is False
    assert summary["decision_gate"]["stage2_stage3_authorized_by_this_gate"] is False


def test_summary_gate_rejects_identity_without_pixel_localization():
    reports = _factorial_reports()
    standard = reports["B"]["prediction_records"]["standard"]
    for record in standard:
        for slot_predictions in record["pred_xy"]:
            for point in slot_predictions:
                point[1] += 9
    standard_evaluation = _metrics(standard)
    standard_evaluation["samples"] = len(standard)
    reports["B"]["evaluations"]["standard"] = standard_evaluation

    summary = summarize(reports, bootstrap_samples=250, seed=17)
    checks = summary["decision_gate"]["checks"]

    assert summary["cells"]["B"]["standard"]["anchor_identity_accuracy"]["estimate"] == 1.0
    assert summary["cells"]["B"]["standard"]["conditional_pck8"]["estimate"] == 0.0
    assert summary["cells"]["B"]["standard"]["true_joint_pck8"]["estimate"] == 0.0
    assert checks["B_standard_conditional_PCK8_at_least_0_30"]["passed"] is False
    assert checks["B_standard_true_joint_PCK8_at_least_0_10"]["passed"] is False
    assert summary["decision_gate"]["overall_passed"] is False
    assert summary["decision_gate"]["stage2_stage3_authorized_by_this_gate"] is False


@pytest.mark.parametrize(
    ("path", "value", "match"),
    [
        (("contract",), "wrong routing contract", "paired swap contract mismatch"),
        (
            ("untargeted", "mean_visibility_l1"),
            1e-12,
            "changed untargeted route mean_visibility_l1",
        ),
    ],
)
def test_summary_rejects_invalid_single_swap_routing_fields(path, value, match):
    reports = _factorial_reports()
    paired = reports["C"]["evaluations"]["single-anchor-swap"]["paired_output_change_vs_standard"]
    target = paired
    for key in path[:-1]:
        target = target[key]
    target[path[-1]] = value

    with pytest.raises(RuntimeError, match=match):
        summarize(reports, bootstrap_samples=10)

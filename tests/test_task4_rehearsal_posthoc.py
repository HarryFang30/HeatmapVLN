from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import pytest
import torch
from torch import nn

from scripts.tools.diagnose_heatmap_shortcuts import (
    heatmap_head_state_dict,
    state_hash,
)
from scripts.tools.evaluate_task4_rehearsal_posthoc import (
    _load_exact_canonical_lora_state,
    generation_coverage,
    load_pilot_state,
    resolve_contracted_selection,
    selection_prefix_contract,
    teacher_forced_coverage,
    validate_head_only_lora_contract,
    validate_pilot_schema,
)
from scripts.tools.train_heatmap_joint_pilot import (
    generic_selection_contract,
    lora_named_parameters,
    select_indices_from_scenes,
)


def _pilot_report(*, mode: str = "joint-rehearsal", step: int = 500, lora_count: int = 1):
    return {
        "task": "task4_joint_pilot",
        "mode": mode,
        "train_steps": step,
        "contract": {
            "final_head_hash": "head-hash",
            "final_lora_hash": "lora-hash",
            "initial_lora_hash": "lora-hash",
            "all_lora_tensors": lora_count,
            "trainable_lora_tensors": 0 if mode == "head-only" else lora_count,
            "trainable_lora_layers": [] if mode == "head-only" else [0],
            "sft_dataset": {},
            "sft_scene_partition": {},
            "sft_rehearsal": {},
            "sft_retention": {},
        },
    }


def _pilot_checkpoint(*, mode: str = "joint-rehearsal", step: int = 500):
    return {
        "task": "task4_joint_pilot",
        "mode": mode,
        "step": step,
        "head_state_dict": {"head": torch.zeros(1)},
        "lora_state_dict": {"lora": torch.zeros(1)},
    }


def test_validate_pilot_schema_accepts_completed_bc_pair():
    result = validate_pilot_schema(
        _pilot_report(),
        _pilot_checkpoint(),
        expected_mode="joint-rehearsal",
    )
    assert result == {
        "mode": "joint-rehearsal",
        "step": 500,
        "head_tensor_count": 1,
        "lora_tensor_count": 1,
        "head_only_base_reference": None,
    }


@pytest.mark.parametrize(
    ("report", "checkpoint", "message"),
    [
        (_pilot_report(mode="unsupported"), _pilot_checkpoint(mode="unsupported"), "Unsupported"),
        (_pilot_report(), _pilot_checkpoint(step=400), "not the report's final step"),
        (_pilot_report(), _pilot_checkpoint(mode="heatmap-lora"), "mode mismatch"),
    ],
)
def test_validate_pilot_schema_rejects_wrong_branch_or_step(report, checkpoint, message):
    with pytest.raises(RuntimeError, match=message):
        validate_pilot_schema(report, checkpoint)


def test_validate_pilot_schema_accepts_strict_head_only_base_reference():
    result = validate_pilot_schema(
        _pilot_report(mode="head-only"),
        _pilot_checkpoint(mode="head-only"),
        expected_mode="head-only",
    )
    assert result["mode"] == "head-only"
    reference = result["head_only_base_reference"]
    assert reference["expected_base_lora_hash"] == "lora-hash"
    assert all(reference["checks"].values())


def test_head_only_reference_rejects_lora_drift_or_wrong_loaded_base_hash():
    contract = _pilot_report(mode="head-only")["contract"]
    drifted = dict(contract, final_lora_hash="changed-lora-hash")
    with pytest.raises(RuntimeError, match="Head-only base LoRA contract failed"):
        validate_head_only_lora_contract(drifted)

    with pytest.raises(RuntimeError, match="Head-only base LoRA contract failed"):
        validate_head_only_lora_contract(contract, base_lora_hash="wrong-base-hash")

    result = validate_head_only_lora_contract(contract, base_lora_hash="lora-hash")
    assert result["checks"]["base_hash_matches_report"]


class _Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.zeros(2, 2))
        self.lora_B = nn.Parameter(torch.zeros(2, 2))


class _Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_Projection(), _Projection()])


class _Head(nn.Module):
    def __init__(self, qwen: nn.Module):
        super().__init__()
        # Reproduce the real lazy HeatmapVLN registration: state_dict exposes
        # a second alias, while named_parameters de-duplicates the physical
        # Qwen LoRA parameters.
        self.qwen = qwen
        self.decoder = nn.Linear(2, 1)


class _Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen2_5_vl = _Backbone()
        self.heatmap_vln = _Head(self.qwen2_5_vl)


def test_load_pilot_state_strictly_overlays_all_lora_and_head_tensors():
    model = _Model()
    assert len(lora_named_parameters(model)) == 4
    assert len([name for name in model.state_dict() if "lora_" in name]) == 8
    lora_state = {
        name: torch.full_like(parameter, 3.0)
        for name, parameter in lora_named_parameters(model).items()
    }
    head_state = {
        name: torch.full_like(value, 5.0)
        for name, value in heatmap_head_state_dict(model.heatmap_vln).items()
    }
    contract = {
        "final_lora_hash": state_hash(lora_state),
        "final_head_hash": state_hash(head_state),
    }
    result = load_pilot_state(
        model,
        {
            "lora_state_dict": lora_state,
            "head_state_dict": head_state,
        },
        contract,
        checkpoint_path="pilot.pth",
    )
    assert result["matched_lora_tensors"] == 4
    assert result["loaded_lora_tensors"] == 4
    assert result["loaded_head_tensors"] == 2
    assert all(result["hash_checks"].values())
    assert all(
        torch.equal(parameter, lora_state[name])
        for name, parameter in lora_named_parameters(model).items()
    )
    assert all(
        torch.equal(value, head_state[name])
        for name, value in heatmap_head_state_dict(model.heatmap_vln).items()
    )


def test_canonical_lora_overlay_still_rejects_real_schema_mismatches():
    model = _Model()
    canonical = {
        name: torch.full_like(parameter, 3.0)
        for name, parameter in lora_named_parameters(model).items()
    }
    first_name = next(iter(canonical))

    missing = dict(canonical)
    missing.pop(first_name)
    with pytest.raises(RuntimeError, match=r"missing=1 unexpected=0 shape_mismatches=0"):
        _load_exact_canonical_lora_state(
            model,
            missing,
            checkpoint_path="missing.pth",
        )

    unexpected = dict(canonical)
    unexpected["not_a_physical_lora.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match=r"missing=0 unexpected=1 shape_mismatches=0"):
        _load_exact_canonical_lora_state(
            model,
            unexpected,
            checkpoint_path="unexpected.pth",
        )

    wrong_shape = dict(canonical)
    wrong_shape[first_name] = torch.zeros(1)
    with pytest.raises(RuntimeError, match=r"missing=0 unexpected=0 shape_mismatches=1"):
        _load_exact_canonical_lora_state(
            model,
            wrong_shape,
            checkpoint_path="shape.pth",
        )


class _SFTDataset:
    def __init__(self, root: Path):
        self.root = root
        self.clips = [
            root / "scene_a" / "clip_1",
            root / "scene_b" / "clip_2",
            root / "scene_c" / "clip_3",
        ]
        self.sample_index = [
            (0, 5), (0, 6), (0, 7), (0, 7),
            (1, 5), (1, 6), (1, 7), (1, 7),
            (2, 5), (2, 6), (2, 7), (2, 7),
        ]
        self._clip_valid_frames = {
            0: [5, 6, 7],
            1: [5, 6, 7],
            2: [5, 6, 7],
        }


def test_resolve_contracted_selection_reconstructs_ordered_indices_and_hash(tmp_path):
    dataset = _SFTDataset(tmp_path)
    source_indices = select_indices_from_scenes(
        dataset,
        ["scene_a", "scene_b"],
        limit=6,
    )
    source_contract = generic_selection_contract(dataset, source_indices)
    indices, contract = resolve_contracted_selection(
        dataset,
        ["scene_a", "scene_b"],
        source_contract,
        label="rehearsal",
    )
    assert indices == source_indices
    assert contract["dataset_indices"] == source_indices
    assert contract["source_contract_exact_match"] is True
    assert contract["sample_identity_sha256"] == source_contract["sample_identity_sha256"]

    tampered = deepcopy(source_contract)
    tampered["sample_identity_sha256"] = "not-the-source-hash"
    with pytest.raises(RuntimeError, match="contract mismatch"):
        resolve_contracted_selection(
            dataset,
            ["scene_a", "scene_b"],
            tampered,
            label="rehearsal",
        )


def test_generation_prefix_has_an_independent_auditable_contract(tmp_path):
    dataset = _SFTDataset(tmp_path)
    indices = select_indices_from_scenes(
        dataset,
        ["scene_a", "scene_b", "scene_c"],
        limit=8,
    )
    selected, contract = selection_prefix_contract(
        dataset,
        indices,
        4,
        label="rehearsal",
    )
    assert selected == indices[:4]
    assert contract["dataset_indices"] == indices[:4]
    assert contract["sample_count"] == 4
    assert contract["is_ordered_prefix_of_contracted_selection"] is True


def test_coverage_requires_ordered_full_records_and_no_generation_errors():
    indices = [4, 8]
    ce = {
        "samples": 2,
        "label_tokens": 5,
        "records": [
            {"dataset_index": 4, "label_tokens": 2},
            {"dataset_index": 8, "label_tokens": 3},
        ],
    }
    assert teacher_forced_coverage(ce, indices)["complete"]
    ce["records"].reverse()
    assert not teacher_forced_coverage(ce, indices)["complete"]

    generation = {
        "complete_coverage": True,
        "requested_samples": 2,
        "attempted_samples": 2,
        "samples": 2,
        "errors": 0,
        "skipped_no_target": 0,
        "records": [{"dataset_index": 4}, {"dataset_index": 8}],
    }
    assert generation_coverage(generation, indices)["complete"]
    generation["errors"] = 1
    assert not generation_coverage(generation, indices)["complete"]

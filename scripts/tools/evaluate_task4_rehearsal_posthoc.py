#!/usr/bin/env python3
"""Post-hoc S1-S2 evaluation for an existing Task-4 pilot checkpoint.

Task-4 training reports contract two disjoint SFT subsets, but the training
script only evaluates the held-out subset.  This utility restores the model
from the original Stage1-S2 checkpoint plus a Task-4 A/B/C pilot checkpoint and
evaluates the *exact contracted rehearsal subset* after training.  It can also
repeat the contracted holdout evaluation in the same run.

The pilot checkpoint is deliberately treated as a strict overlay: every LoRA
tensor and every non-Qwen HeatmapVLN tensor must match the checkpoint schema,
and their loaded hashes must equal the final hashes recorded in the pilot
report.  Likewise, the balanced SFT dataset, scene split, ordered identities,
selection hashes, and category counts must reproduce the source report before
any metric is computed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import logging
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tools.diagnose_heatmap_shortcuts import (  # noqa: E402
    heatmap_head_state_dict,
    load_stage1_s2_lora,
    set_seed,
    state_hash,
)
from scripts.tools.train_heatmap_joint_pilot import (  # noqa: E402
    build_sft_dataset_and_collator,
    evaluate_sft_ce,
    evaluate_sft_generation,
    generic_selection_contract,
    load_pilot_config,
    lora_named_parameters,
    select_indices_from_scenes,
    select_scene_partition,
    sft_dataset_contract,
)
from scripts.training import (  # noqa: E402
    build_model,
    safe_torch_load,
)


LOGGER = logging.getLogger("task4_rehearsal_posthoc")
PILOT_MODES = ("head-only", "heatmap-lora", "joint-rehearsal")
SELECTION_CONTRACT_KEYS = (
    "sample_count",
    "sample_identity_sha256",
    "sample_identities",
    "scenes",
    "category_counts",
)
DATASET_CONTRACT_KEYS = (
    "clip_count",
    "scene_count",
    "scenes",
    "per_scene_clip_counts",
    "clip_identity_sha256",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--base-checkpoint",
        required=True,
        help="Original Stage1-S2 checkpoint used to launch the Task-4 pilot.",
    )
    parser.add_argument(
        "--pilot-checkpoint",
        required=True,
        help="Task-4 A/B/C checkpoint_final.pth (or an equivalent completed checkpoint).",
    )
    parser.add_argument(
        "--pilot-report",
        required=True,
        help="report.json emitted by the same Task-4 pilot branch.",
    )
    parser.add_argument("--output", required=True, help="Destination JSON report path.")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--expected-mode",
        choices=PILOT_MODES,
        default=None,
        help="Optional guard against evaluating the wrong A/B/C checkpoint.",
    )
    parser.add_argument(
        "--rehearsal-generation-samples",
        type=int,
        choices=(64, 128),
        default=128,
        help="Autoregressive generation on a deterministic prefix of the contracted 128 samples.",
    )
    parser.add_argument(
        "--evaluate-holdout",
        action="store_true",
        help="Also repeat pooled CE and generation on the contracted holdout subset.",
    )
    parser.add_argument("--holdout-generation-samples", type=int, default=64)
    parser.add_argument("--max-new-tokens", type=int, default=16)
    parser.add_argument("--coord-tolerance", type=float, default=15.0)
    return parser.parse_args()


def _json_load(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise TypeError(f"Expected a JSON object: {path}")
    return payload


def _json_dump(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=True)
    temporary.replace(target)


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _resolved(path: str | Path) -> str:
    return str(Path(path).expanduser().resolve())


def _require_mapping(mapping: dict[str, Any], key: str, *, source: str) -> dict[str, Any]:
    value = mapping.get(key)
    if not isinstance(value, dict):
        raise RuntimeError(f"{source} lacks mapping field {key!r}")
    return value


def validate_pilot_schema(
    report: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    expected_mode: str | None = None,
) -> dict[str, Any]:
    """Validate that a report/checkpoint pair is a completed Task-4 branch."""
    if report.get("task") != "task4_joint_pilot":
        raise RuntimeError(f"Unexpected pilot report task={report.get('task')!r}")
    if checkpoint.get("task") != "task4_joint_pilot":
        raise RuntimeError(f"Unexpected pilot checkpoint task={checkpoint.get('task')!r}")

    report_mode = report.get("mode")
    checkpoint_mode = checkpoint.get("mode")
    if report_mode not in PILOT_MODES:
        raise RuntimeError(f"Unsupported Task-4 pilot mode {report_mode!r}")
    if checkpoint_mode != report_mode:
        raise RuntimeError(
            f"Pilot report/checkpoint mode mismatch: report={report_mode!r} "
            f"checkpoint={checkpoint_mode!r}"
        )
    if expected_mode is not None and report_mode != expected_mode:
        raise RuntimeError(
            f"Expected pilot mode {expected_mode!r}, loaded {report_mode!r}"
        )

    report_steps = int(report.get("train_steps", -1))
    checkpoint_step = int(checkpoint.get("step", -2))
    if checkpoint_step != report_steps:
        raise RuntimeError(
            f"Pilot checkpoint is not the report's final step: "
            f"checkpoint={checkpoint_step} report={report_steps}"
        )

    report_contract = _require_mapping(report, "contract", source="pilot report")
    for key in (
        "final_head_hash",
        "final_lora_hash",
        "all_lora_tensors",
        "sft_dataset",
        "sft_scene_partition",
        "sft_rehearsal",
        "sft_retention",
    ):
        if key not in report_contract:
            raise RuntimeError(f"Pilot report contract lacks {key!r}")

    head_state = checkpoint.get("head_state_dict")
    lora_state = checkpoint.get("lora_state_dict")
    if not isinstance(head_state, dict) or not head_state:
        raise RuntimeError("Pilot checkpoint has no non-empty head_state_dict")
    if not isinstance(lora_state, dict) or not lora_state:
        raise RuntimeError("Pilot checkpoint has no non-empty lora_state_dict")
    if any(not isinstance(value, torch.Tensor) for value in head_state.values()):
        raise RuntimeError("Pilot head_state_dict contains a non-tensor value")
    if any(not isinstance(value, torch.Tensor) for value in lora_state.values()):
        raise RuntimeError("Pilot lora_state_dict contains a non-tensor value")

    expected_lora_count = int(report_contract["all_lora_tensors"])
    if len(lora_state) != expected_lora_count:
        raise RuntimeError(
            f"Pilot LoRA tensor count mismatch: checkpoint={len(lora_state)} "
            f"report={expected_lora_count}"
        )
    head_only_lora = None
    if report_mode == "head-only":
        head_only_lora = validate_head_only_lora_contract(report_contract)
    return {
        "mode": report_mode,
        "step": checkpoint_step,
        "head_tensor_count": len(head_state),
        "lora_tensor_count": len(lora_state),
        "head_only_base_reference": head_only_lora,
    }


def validate_head_only_lora_contract(
    report_contract: dict[str, Any],
    *,
    base_lora_hash: str | None = None,
) -> dict[str, Any]:
    """Require branch A to be an exact, unmodified Stage1-S2 LoRA reference."""
    initial_hash = report_contract.get("initial_lora_hash")
    final_hash = report_contract.get("final_lora_hash")
    trainable_tensors = int(report_contract.get("trainable_lora_tensors", -1))
    trainable_layers = report_contract.get("trainable_lora_layers")
    checks = {
        "initial_hash_present": isinstance(initial_hash, str) and bool(initial_hash),
        "initial_equals_final": initial_hash == final_hash,
        "trainable_lora_tensors_zero": trainable_tensors == 0,
        "trainable_lora_layers_empty": trainable_layers == [],
        "base_hash_matches_report": (
            True if base_lora_hash is None else base_lora_hash == initial_hash
        ),
    }
    if not all(checks.values()):
        raise RuntimeError(f"Head-only base LoRA contract failed: {checks}")
    return {
        "expected_base_lora_hash": initial_hash,
        "checks": checks,
    }


def _load_exact_head_state(
    module: torch.nn.Module,
    state: dict[str, torch.Tensor],
) -> int:
    expected = heatmap_head_state_dict(module)
    missing = sorted(set(expected) - set(state))
    unexpected = sorted(set(state) - set(expected))
    shape_mismatches = sorted(
        name
        for name in set(expected) & set(state)
        if tuple(expected[name].shape) != tuple(state[name].shape)
    )
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            "Incompatible Task-4 heatmap head: "
            f"missing={missing[:5]} unexpected={unexpected[:5]} "
            f"shape_mismatches={shape_mismatches[:5]}"
        )
    load_missing, load_unexpected = module.load_state_dict(state, strict=False)
    non_qwen_missing = [name for name in load_missing if not name.startswith("qwen.")]
    if non_qwen_missing or load_unexpected:
        raise RuntimeError(
            "Task-4 head load was incomplete: "
            f"missing={non_qwen_missing[:5]} unexpected={list(load_unexpected)[:5]}"
        )
    return len(state)


def _load_exact_canonical_lora_state(
    model: torch.nn.Module,
    state: dict[str, torch.Tensor],
    *,
    checkpoint_path: str,
) -> int:
    """Load one tensor per physical LoRA parameter, ignoring module aliases.

    Once HeatmapVLN is materialised, ``model.state_dict()`` exposes the same
    Qwen parameters through both ``qwen2_5_vl`` and ``heatmap_vln.qwen``.  The
    latter is a registered module alias, not another adapter.  In contrast,
    ``named_parameters()`` removes duplicate parameter objects and is also the
    canonical namespace used by ``full_lora_state`` when Task-4 checkpoints
    are saved.  Validate and copy against that exact physical mapping so an
    alias cannot create false missing keys or conceal a real mismatch.
    """
    expected = lora_named_parameters(model)
    expected_keys = set(expected)
    checkpoint_keys = set(state)
    missing = sorted(expected_keys - checkpoint_keys)
    unexpected = sorted(checkpoint_keys - expected_keys)
    shape_mismatches = sorted(
        name
        for name in expected_keys & checkpoint_keys
        if tuple(expected[name].shape) != tuple(state[name].shape)
    )
    if missing or unexpected or shape_mismatches:
        raise RuntimeError(
            "Incomplete canonical LoRA checkpoint overlay refused from "
            f"{checkpoint_path}: physical_model_lora={len(expected)} "
            f"checkpoint_lora={len(state)} missing={len(missing)} "
            f"unexpected={len(unexpected)} shape_mismatches={len(shape_mismatches)} "
            f"missing_preview={missing[:5]} unexpected_preview={unexpected[:5]} "
            f"shape_mismatch_preview={shape_mismatches[:5]}"
        )
    with torch.no_grad():
        for name, parameter in expected.items():
            parameter.copy_(state[name].to(device=parameter.device, dtype=parameter.dtype))
    return len(expected)


def load_pilot_state(
    model: torch.nn.Module,
    checkpoint: dict[str, Any],
    report_contract: dict[str, Any],
    *,
    checkpoint_path: str,
) -> dict[str, Any]:
    """Strictly overlay every pilot LoRA and heatmap-head tensor."""
    lora_state = checkpoint["lora_state_dict"]
    head_state = checkpoint["head_state_dict"]
    loaded_lora = _load_exact_canonical_lora_state(
        model,
        lora_state,
        checkpoint_path=checkpoint_path,
    )
    loaded_head = _load_exact_head_state(model.heatmap_vln, head_state)

    loaded_lora_state = {
        name: parameter.detach().cpu().clone()
        for name, parameter in lora_named_parameters(model).items()
    }
    loaded_head_state = heatmap_head_state_dict(model.heatmap_vln)
    checkpoint_lora_hash = state_hash(lora_state)
    checkpoint_head_hash = state_hash(head_state)
    loaded_lora_hash = state_hash(loaded_lora_state)
    loaded_head_hash = state_hash(loaded_head_state)

    expected_lora_hash = str(report_contract["final_lora_hash"])
    expected_head_hash = str(report_contract["final_head_hash"])
    hash_checks = {
        "checkpoint_lora_matches_report": checkpoint_lora_hash == expected_lora_hash,
        "checkpoint_head_matches_report": checkpoint_head_hash == expected_head_hash,
        "loaded_lora_matches_checkpoint": loaded_lora_hash == checkpoint_lora_hash,
        "loaded_head_matches_checkpoint": loaded_head_hash == checkpoint_head_hash,
    }
    if not all(hash_checks.values()):
        raise RuntimeError(f"Task-4 checkpoint hash contract failed: {hash_checks}")
    return {
        "matched_lora_tensors": loaded_lora,
        "loaded_lora_tensors": loaded_lora,
        "loaded_head_tensors": loaded_head,
        "checkpoint_lora_hash": checkpoint_lora_hash,
        "checkpoint_head_hash": checkpoint_head_hash,
        "loaded_lora_hash": loaded_lora_hash,
        "loaded_head_hash": loaded_head_hash,
        "hash_checks": hash_checks,
    }


def _assert_contract_fields_equal(
    actual: dict[str, Any],
    expected: dict[str, Any],
    keys: tuple[str, ...],
    *,
    label: str,
) -> None:
    mismatches = {
        key: {"actual": actual.get(key), "expected": expected.get(key)}
        for key in keys
        if actual.get(key) != expected.get(key)
    }
    if mismatches:
        preview = {
            key: value
            for key, value in mismatches.items()
            if key not in {"sample_identities", "scenes", "per_scene_clip_counts"}
        }
        if not preview:
            preview = {key: "ordered values differ" for key in mismatches}
        raise RuntimeError(f"{label} contract mismatch: {preview}")


def resolve_contracted_selection(
    dataset: Any,
    scenes: list[str],
    expected_contract: dict[str, Any],
    *,
    label: str,
) -> tuple[list[int], dict[str, Any]]:
    """Reconstruct and verify the ordered dataset indices used by Task-4."""
    sample_count = int(expected_contract.get("sample_count", 0))
    if sample_count <= 0:
        raise RuntimeError(f"{label} source contract has invalid sample_count={sample_count}")
    indices = select_indices_from_scenes(dataset, scenes, sample_count)
    actual_contract = generic_selection_contract(dataset, indices)
    _assert_contract_fields_equal(
        actual_contract,
        expected_contract,
        SELECTION_CONTRACT_KEYS,
        label=label,
    )
    actual_contract["dataset_indices"] = [int(index) for index in indices]
    actual_contract["source_contract_exact_match"] = True
    return indices, actual_contract


def selection_prefix_contract(
    dataset: Any,
    indices: list[int],
    limit: int,
    *,
    label: str,
) -> tuple[list[int], dict[str, Any]]:
    if limit <= 0 or limit > len(indices):
        raise ValueError(f"{label} generation sample count must be in [1, {len(indices)}], got {limit}")
    selected = indices[:limit]
    contract = generic_selection_contract(dataset, selected)
    contract["dataset_indices"] = [int(index) for index in selected]
    contract["is_ordered_prefix_of_contracted_selection"] = True
    return selected, contract


def teacher_forced_coverage(metrics: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    records = metrics.get("records", [])
    record_indices = [record.get("dataset_index") for record in records]
    complete = bool(
        metrics.get("samples") == len(indices)
        and len(records) == len(indices)
        and record_indices == indices
        and int(metrics.get("label_tokens", 0)) > 0
        and all(int(record.get("label_tokens", 0)) > 0 for record in records)
    )
    return {
        "requested_samples": len(indices),
        "evaluated_samples": int(metrics.get("samples", 0)),
        "record_count": len(records),
        "label_tokens": int(metrics.get("label_tokens", 0)),
        "ordered_index_coverage": record_indices == indices,
        "complete": complete,
    }


def generation_coverage(metrics: dict[str, Any], indices: list[int]) -> dict[str, Any]:
    records = metrics.get("records", [])
    record_indices = [record.get("dataset_index") for record in records]
    complete = bool(
        metrics.get("complete_coverage") is True
        and int(metrics.get("requested_samples", -1)) == len(indices)
        and int(metrics.get("attempted_samples", -1)) == len(indices)
        and int(metrics.get("samples", -1)) == len(indices)
        and int(metrics.get("errors", -1)) == 0
        and int(metrics.get("skipped_no_target", -1)) == 0
        and len(records) == len(indices)
        and record_indices == indices
    )
    return {
        "requested_samples": len(indices),
        "attempted_samples": int(metrics.get("attempted_samples", 0)),
        "evaluated_samples": int(metrics.get("samples", 0)),
        "record_count": len(records),
        "errors": int(metrics.get("errors", 0)),
        "skipped_no_target": int(metrics.get("skipped_no_target", 0)),
        "ordered_index_coverage": record_indices == indices,
        "complete": complete,
    }


def evaluate_selection(
    *,
    model: torch.nn.Module,
    dataset: Any,
    collator: Any,
    cfg: dict[str, Any],
    indices: list[int],
    generation_indices: list[int],
    generation_args: argparse.Namespace,
    device: torch.device,
    label: str,
) -> dict[str, Any]:
    LOGGER.info("Evaluating %s pooled teacher-forced CE on %d samples", label, len(indices))
    teacher_forced = evaluate_sft_ce(model, dataset, collator, indices, device)
    LOGGER.info("Evaluating %s generation on %d samples", label, len(generation_indices))
    generation = evaluate_sft_generation(
        model,
        dataset,
        generation_indices,
        cfg,
        generation_args,
        device,
    )
    coverage = {
        "teacher_forced": teacher_forced_coverage(teacher_forced, indices),
        "generation": generation_coverage(generation, generation_indices),
    }
    if not coverage["teacher_forced"]["complete"]:
        raise RuntimeError(f"{label} teacher-forced evaluation coverage is incomplete")
    if not coverage["generation"]["complete"]:
        raise RuntimeError(f"{label} generation evaluation coverage is incomplete")
    return {
        "teacher_forced": teacher_forced,
        "generation": generation,
        "coverage": coverage,
    }


def _validate_source_paths(
    report: dict[str, Any],
    checkpoint: dict[str, Any],
    *,
    base_checkpoint: str,
) -> None:
    expected_base = _resolved(base_checkpoint)
    sources = {
        "pilot report checkpoint": report.get("checkpoint"),
        "pilot checkpoint args.checkpoint": checkpoint.get("args", {}).get("checkpoint"),
    }
    mismatches = {
        label: value
        for label, value in sources.items()
        if value is None or _resolved(value) != expected_base
    }
    if mismatches:
        raise RuntimeError(
            f"Stage1-S2 base checkpoint does not match the pilot sources: "
            f"requested={expected_base} mismatches={mismatches}"
        )


def main() -> int:
    args = parse_args()
    started_at = time.time()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    pilot_report_path = _resolved(args.pilot_report)
    pilot_checkpoint_path = _resolved(args.pilot_checkpoint)
    base_checkpoint_path = _resolved(args.base_checkpoint)
    report = _json_load(pilot_report_path)
    checkpoint = safe_torch_load(pilot_checkpoint_path, trust_checkpoint=True)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a checkpoint mapping: {pilot_checkpoint_path}")
    schema = validate_pilot_schema(report, checkpoint, expected_mode=args.expected_mode)
    _validate_source_paths(report, checkpoint, base_checkpoint=base_checkpoint_path)

    report_contract = report["contract"]
    pilot_args = checkpoint.get("args", {})
    seed = int(report.get("seed", pilot_args.get("seed", 42)))
    num_history = int(pilot_args.get("num_history", 2))
    config_args = SimpleNamespace(
        mode="head-only",
        config=report["config"],
        data_root=report["data_root"],
        device=args.device,
        num_history=num_history,
    )
    cfg = load_pilot_config(config_args)
    device = torch.device(args.device)

    set_seed(seed)
    model = build_model(
        cfg,
        verbose=True,
        device=args.device,
        enable_action_head=False,
    )
    model.qwen2_5_vl._load_model()
    base_load = load_stage1_s2_lora(model, base_checkpoint_path)
    base_lora_hash = state_hash(
        {
            name: parameter.detach().cpu().clone()
            for name, parameter in lora_named_parameters(model).items()
        }
    )
    if schema["mode"] == "head-only":
        schema["head_only_base_reference"] = validate_head_only_lora_contract(
            report_contract,
            base_lora_hash=base_lora_hash,
        )
    set_seed(seed + 991)
    model._ensure_heatmap_vln()
    pilot_load = load_pilot_state(
        model,
        checkpoint,
        report_contract,
        checkpoint_path=pilot_checkpoint_path,
    )
    # The custom checkpoint also carries a large optimizer state.  None of it
    # is needed after the strict tensor overlay, so release the CPU payload
    # before materialising/evaluating SFT samples.
    del checkpoint
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    dataset_args = SimpleNamespace(
        sft_config=report["sft_config"],
        sft_data_root=report["sft_data_root"],
        sft_max_clips=0,
    )
    dataset, collator, sft_cfg = build_sft_dataset_and_collator(dataset_args, model)
    current_dataset_contract = sft_dataset_contract(dataset)
    source_dataset_contract = report_contract["sft_dataset"]
    _assert_contract_fields_equal(
        current_dataset_contract,
        source_dataset_contract,
        DATASET_CONTRACT_KEYS,
        label="SFT dataset",
    )

    source_partition = report_contract["sft_scene_partition"]
    requested_holdout = int(source_partition["requested_holdout_scene_count"])
    rehearsal_scenes, holdout_scenes = select_scene_partition(
        dataset,
        seed=seed,
        holdout_scene_count=requested_holdout,
    )
    if rehearsal_scenes != source_partition["rehearsal_scenes"]:
        raise RuntimeError("Reconstructed rehearsal scenes differ from the pilot report")
    if holdout_scenes != source_partition["holdout_scenes"]:
        raise RuntimeError("Reconstructed holdout scenes differ from the pilot report")

    rehearsal_indices, rehearsal_contract = resolve_contracted_selection(
        dataset,
        rehearsal_scenes,
        report_contract["sft_rehearsal"],
        label="SFT rehearsal",
    )
    holdout_indices, holdout_contract = resolve_contracted_selection(
        dataset,
        holdout_scenes,
        report_contract["sft_retention"],
        label="SFT holdout",
    )
    source_holdout_records = (
        report.get("sft_retention", {})
        .get("teacher_forced_after", {})
        .get("records", [])
    )
    if source_holdout_records:
        source_holdout_indices = [int(record["dataset_index"]) for record in source_holdout_records]
        if source_holdout_indices != holdout_indices:
            raise RuntimeError("Reconstructed holdout dataset indices differ from pilot records")

    rehearsal_generation_indices, rehearsal_generation_contract = selection_prefix_contract(
        dataset,
        rehearsal_indices,
        args.rehearsal_generation_samples,
        label="rehearsal",
    )
    holdout_generation_indices: list[int] = []
    holdout_generation_contract = None
    if args.evaluate_holdout:
        holdout_generation_indices, holdout_generation_contract = selection_prefix_contract(
            dataset,
            holdout_indices,
            args.holdout_generation_samples,
            label="holdout",
        )

    generation_args = SimpleNamespace(
        max_new_tokens=args.max_new_tokens,
        coord_tolerance=args.coord_tolerance,
    )
    rehearsal_evaluation = evaluate_selection(
        model=model,
        dataset=dataset,
        collator=collator,
        cfg=sft_cfg,
        indices=rehearsal_indices,
        generation_indices=rehearsal_generation_indices,
        generation_args=generation_args,
        device=device,
        label="contracted rehearsal",
    )
    holdout_evaluation = None
    if args.evaluate_holdout:
        holdout_evaluation = evaluate_selection(
            model=model,
            dataset=dataset,
            collator=collator,
            cfg=sft_cfg,
            indices=holdout_indices,
            generation_indices=holdout_generation_indices,
            generation_args=generation_args,
            device=device,
            label="contracted holdout",
        )

    output = {
        "task": "task4_posthoc_rehearsal_evaluation",
        "pilot_mode": schema["mode"],
        "pilot_step": schema["step"],
        "seed": seed,
        "device": args.device,
        "elapsed_seconds": time.time() - started_at,
        "sources": {
            "base_checkpoint": base_checkpoint_path,
            "pilot_checkpoint": pilot_checkpoint_path,
            "pilot_report": pilot_report_path,
            "pilot_report_sha256": _file_sha256(pilot_report_path),
            "config": _resolved(report["config"]),
            "sft_config": _resolved(report["sft_config"]),
            "sft_data_root": _resolved(report["sft_data_root"]),
        },
        "checkpoint_restore": {
            "schema": schema,
            "base_load": base_load,
            "base_lora_hash_before_pilot_overlay": base_lora_hash,
            "pilot_overlay": pilot_load,
        },
        "metric_protocol": {
            "teacher_forced_pooling": "label-token-weighted mean cross entropy",
            "generation_subset": "ordered prefix of the exact contracted selection",
            "rehearsal_generation_samples": len(rehearsal_generation_indices),
            "holdout_generation_samples": len(holdout_generation_indices),
            "max_new_tokens": args.max_new_tokens,
            "coord_tolerance": args.coord_tolerance,
        },
        "contract": {
            "dataset_exact_match": True,
            "sft_dataset": current_dataset_contract,
            "scene_partition_exact_match": True,
            "sft_scene_partition": {
                "requested_holdout_scene_count": requested_holdout,
                "rehearsal_scenes": rehearsal_scenes,
                "holdout_scenes": holdout_scenes,
            },
            "sft_rehearsal": rehearsal_contract,
            "sft_rehearsal_generation": rehearsal_generation_contract,
            "sft_holdout": holdout_contract,
            "sft_holdout_generation": holdout_generation_contract,
        },
        "evaluation": {
            "rehearsal": rehearsal_evaluation,
            "holdout": holdout_evaluation,
        },
        "pilot_holdout_after_reference": report.get("sft_retention", {}),
    }
    _json_dump(args.output, output)
    LOGGER.info("Wrote post-hoc report to %s", _resolved(args.output))
    print(json.dumps(output, indent=2, ensure_ascii=False, allow_nan=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Derive a smaller strict Task-3.6 selection from an existing manifest.

This tool never revisits poses, depth, or candidate generation.  Its entire
candidate pool is the already-verified records in a parent Task-3.6 manifest.
Consequently it can reduce a requested split size without relaxing any
per-record or dataset-level selection constraint.

The output directory is published by one atomic directory rename.  Refusing
to overwrite an existing directory also prevents a reader from observing a
new manifest alongside stale JSONL files (or the reverse).
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.tools.build_multi_history_selection import (
    VIEW_NAMES,
    assert_scene_disjoint,
    audit_selection,
    deterministic_balanced_selection,
    failure_audit,
    selection_manifest,
)

from src.data.explicit_multi_history import (
    canonical_sha256,
    load_multi_history_records,
)

MANIFEST_NAME = "multi_history_selection_manifest.json"
SPLITS = ("train", "val")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Derive a strict smaller train/val subset from a Task-3.6 manifest.")
    parser.add_argument("--parent-manifest", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--train-samples", type=int, default=128)
    parser.add_argument("--val-samples", type=int, default=40)
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Subset tie-break seed; defaults to the parent's selection seed.",
    )
    return parser.parse_args()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _target_view_counts(records: list[dict[str, Any]]) -> Counter[str]:
    counts: Counter[str] = Counter()
    for record in records:
        for anchor in record["label_metadata"]["anchors"]:
            primary = anchor.get("primary_target")
            if primary is not None:
                counts[str(primary["view"])] += 1
    return counts


def _preserved_selection_diagnostics(
    records: list[dict[str, Any]],
    *,
    requested_samples: int,
    num_history: int,
    min_target_view_fraction: float,
    max_target_view_fraction: float,
) -> dict[str, Any]:
    """Audit an unchanged full parent split without running the greedy selector."""
    event_count = requested_samples * num_history
    per_view_floor = math.ceil(min_target_view_fraction * event_count - 1e-12)
    per_view_cap = math.floor(max_target_view_fraction * event_count + 1e-12)
    counts = _target_view_counts(records)
    actual_events = sum(counts.values())
    fractions = {view: (float(counts[view]) / actual_events if actual_events else 0.0) for view in VIEW_NAMES}
    unmet: list[str] = []
    if len(records) != requested_samples:
        unmet.append(f"sample_count_mismatch:{len(records)}!={requested_samples}")
    if actual_events != event_count:
        unmet.append(f"target_event_count_mismatch:{actual_events}!={event_count}")
    if len(records) == requested_samples and actual_events == event_count:
        for view in VIEW_NAMES:
            if fractions[view] + 1e-12 < min_target_view_fraction:
                unmet.append(f"target_view_fraction:{view}:{fractions[view]:.6f}<{min_target_view_fraction:.6f}")
            if fractions[view] > max_target_view_fraction + 1e-12:
                unmet.append(f"target_view_fraction:{view}:{fractions[view]:.6f}>{max_target_view_fraction:.6f}")
    return {
        "requested_samples": requested_samples,
        "pool_samples": len(records),
        "selected_samples": len(records),
        "selection_complete": not unmet,
        "selection_strategy": "preserve_parent_exact_order_and_record_hashes",
        "target_events_at_requested_size": event_count,
        "per_view_event_floor": per_view_floor,
        "per_view_event_cap": per_view_cap,
        "min_target_view_fraction": min_target_view_fraction,
        "max_target_view_fraction": max_target_view_fraction,
        "target_view_counts": {view: int(counts[view]) for view in VIEW_NAMES},
        "target_view_fractions": fractions,
        "scene_counts": dict(sorted(Counter(str(record["scene"]) for record in records).items())),
        "unmet_constraints": unmet,
    }


def _derive_split(
    parent_records: list[dict[str, Any]],
    parent_payload: dict[str, Any],
    *,
    split: str,
    requested_samples: int,
    num_history: int,
    min_target_view_fraction: float,
    max_target_view_fraction: float,
    seed: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    if requested_samples <= 0:
        raise ValueError(f"{split} requested sample count must be positive")
    if requested_samples > len(parent_records):
        raise ValueError(
            f"Cannot derive {requested_samples} {split} samples from only {len(parent_records)} strict parent records"
        )

    parent_order = [str(record["sample_id"]) for record in parent_records]
    parent_hashes = {str(record["sample_id"]): str(record["record_sha256"]) for record in parent_records}
    if len(parent_hashes) != len(parent_records):
        raise ValueError(f"Parent {split} split contains duplicate sample identities")

    if requested_samples == len(parent_records):
        # Do not even invoke the greedy selector in this branch: exact parent
        # ordering is part of the derivation contract.
        selected = list(parent_records)
        balance = _preserved_selection_diagnostics(
            selected,
            requested_samples=requested_samples,
            num_history=num_history,
            min_target_view_fraction=min_target_view_fraction,
            max_target_view_fraction=max_target_view_fraction,
        )
    else:
        selected, balance = deterministic_balanced_selection(
            parent_records,
            requested_samples=requested_samples,
            num_history=num_history,
            min_target_view_fraction=min_target_view_fraction,
            max_target_view_fraction=max_target_view_fraction,
            seed=seed,
        )
        balance = {
            **balance,
            "selection_strategy": "deterministic_balanced_selection_from_parent_records_only",
        }

    parent_identity_set = set(parent_order)
    for record in selected:
        identity = str(record["sample_id"])
        if identity not in parent_identity_set:
            raise RuntimeError(f"Derived {split} record is not in the parent pool: {identity}")
        if str(record["record_sha256"]) != parent_hashes[identity]:
            raise RuntimeError(f"Derived {split} record hash changed: {identity}")

    current_failures: list[dict[str, Any]] = []
    if not bool(balance["selection_complete"]):
        current_failures.append(
            {
                "split": split,
                "stage": "strict_subset_derivation",
                "reason": "balanced_selection_incomplete",
                "details": copy.deepcopy(balance),
            }
        )

    return selected, {
        "selection_manifest": selection_manifest(selected),
        "selection_audit": audit_selection(selected),
        # Candidate generation is not rerun.  Retaining the exact object here
        # keeps provenance available while its scope label prevents it being
        # mistaken for a newly generated pool.
        "candidate_catalog": copy.deepcopy(parent_payload.get("candidate_catalog")),
        "candidate_catalog_scope": "inherited_parent_generation_provenance_not_recomputed",
        "source_inventory": copy.deepcopy(parent_payload.get("source_inventory")),
        "balanced_selector": balance,
        # Only derivation-time failures determine current readiness.
        "failure_audit": failure_audit(current_failures),
        "parent_failure_audit": copy.deepcopy(parent_payload.get("failure_audit")),
        "parent_failure_audit_scope": (
            "historical_parent_candidate_generation_provenance_only; "
            "parent sample-count shortfalls are not current derivation shortfalls"
        ),
        "parent_selection_provenance": {
            "selection_manifest": copy.deepcopy(parent_payload.get("selection_manifest")),
            "selection_audit": copy.deepcopy(parent_payload.get("selection_audit")),
            "balanced_selector": copy.deepcopy(parent_payload.get("balanced_selector")),
            "selection_complete": bool(parent_payload.get("selection_complete", False)),
        },
        "selection_complete": bool(balance["selection_complete"]),
        "records": selected,
    }


def _jsonl_bytes(records: list[dict[str, Any]]) -> bytes:
    rows = [json.dumps(record, ensure_ascii=False, allow_nan=False) + "\n" for record in records]
    return "".join(rows).encode("utf-8")


def _write_file(path: Path, payload: bytes) -> None:
    with path.open("wb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _publish_bundle_atomic(
    output_dir: Path,
    *,
    manifest_bytes: bytes,
    train_bytes: bytes,
    val_bytes: bytes,
) -> None:
    """Publish all three files at once; an existing output is never mutated."""
    output_dir = output_dir.resolve()
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite existing derived output directory: {output_dir}")
    staging = Path(
        tempfile.mkdtemp(
            prefix=f".{output_dir.name}.tmp-",
            dir=output_dir.parent,
        )
    )
    try:
        _write_file(staging / "train_selection.jsonl", train_bytes)
        _write_file(staging / "val_selection.jsonl", val_bytes)
        # The manifest is the final commit marker even inside staging.
        _write_file(staging / MANIFEST_NAME, manifest_bytes)
        os.replace(staging, output_dir)
        directory_fd = os.open(output_dir.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    except Exception:
        if staging.exists():
            shutil.rmtree(staging)
        raise


def derive_multi_history_subset(
    parent_manifest_path: str | Path,
    output_dir: str | Path,
    *,
    train_samples: int = 128,
    val_samples: int = 40,
    seed: int | None = None,
) -> dict[str, Any]:
    """Verify a parent manifest and atomically publish its strict subset."""
    parent_path = Path(parent_manifest_path).resolve()
    if not parent_path.is_file():
        raise FileNotFoundError(f"Parent manifest does not exist: {parent_path}")

    parent_records: dict[str, list[dict[str, Any]]] = {}
    parent: dict[str, Any] | None = None
    for split in SPLITS:
        records, loaded_manifest = load_multi_history_records(parent_path, split)
        if parent is None:
            parent = loaded_manifest
        elif loaded_manifest["manifest_sha256"] != parent["manifest_sha256"]:
            raise RuntimeError("Parent manifest changed while splits were being verified")
        parent_records[split] = records
    assert parent is not None

    parameters = parent.get("selection_parameters")
    if not isinstance(parameters, dict):
        raise ValueError("Parent manifest is missing selection_parameters")
    required_parameters = {
        "num_history",
        "min_target_view_fraction",
        "max_target_view_fraction",
        "seed",
    }
    missing = sorted(required_parameters - set(parameters))
    if missing:
        raise ValueError(f"Parent selection_parameters are missing: {missing}")
    num_history = int(parameters["num_history"])
    min_fraction = float(parameters["min_target_view_fraction"])
    max_fraction = float(parameters["max_target_view_fraction"])
    derivation_seed = int(parameters["seed"] if seed is None else seed)
    requested = {"train": int(train_samples), "val": int(val_samples)}

    split_outputs: dict[str, dict[str, Any]] = {}
    selected_by_split: dict[str, list[dict[str, Any]]] = {}
    for split in SPLITS:
        selected, payload = _derive_split(
            parent_records[split],
            parent["splits"][split],
            split=split,
            requested_samples=requested[split],
            num_history=num_history,
            min_target_view_fraction=min_fraction,
            max_target_view_fraction=max_fraction,
            seed=derivation_seed,
        )
        selected_by_split[split] = selected
        split_outputs[split] = payload

    assert_scene_disjoint(selected_by_split["train"], selected_by_split["val"])
    train_scenes = set(split_outputs["train"]["selection_manifest"]["scenes"])
    val_scenes = set(split_outputs["val"]["selection_manifest"]["scenes"])
    scene_overlap = sorted(train_scenes & val_scenes)
    current_ready = all(bool(split_outputs[split]["selection_complete"]) for split in SPLITS) and not scene_overlap

    train_bytes = _jsonl_bytes(selected_by_split["train"])
    val_bytes = _jsonl_bytes(selected_by_split["val"])
    parent_ready = bool(parent.get("ready", False))
    if current_ready and not parent_ready:
        readiness_reason = (
            "ready because requested counts were reduced to a strictly feasible subset of "
            "already-valid parent records; no record or dataset-level constraint was relaxed. "
            "The parent's ready=false status reflected its original request and is retained "
            "only as provenance."
        )
    elif current_ready:
        readiness_reason = "ready because every requested split is complete under unchanged parent constraints."
    else:
        readiness_reason = (
            "not ready because at least one requested subset could not satisfy the unchanged parent constraints."
        )

    output: dict[str, Any] = {
        "schema_version": parent["schema_version"],
        "derived_from": {
            "path": str(parent_path),
            "file_sha256": _file_sha256(parent_path),
            "manifest_sha256": str(parent["manifest_sha256"]),
            "ready": parent_ready,
            "split_record_counts": {split: len(parent_records[split]) for split in SPLITS},
            "status_scope": (
                "parent readiness and failures are provenance only; current readiness is "
                "recomputed for the derived requested counts"
            ),
        },
        "derivation_parameters": {
            "requested_samples": requested,
            "seed": derivation_seed,
            "seed_source": "parent_selection_parameters" if seed is None else "cli_override",
            "candidate_pool": "parent_splits[*].records_only",
            "constraint_source": "parent.selection_parameters",
            "constraint_overrides": {},
            "constraints_relaxed": False,
            "equal_size_policy": "preserve_parent_exact_order_and_record_hashes",
            "smaller_size_policy": "deterministic_balanced_selection",
        },
        # Copy, rather than reconstruct, every constraint-bearing contract.
        "selection_parameters": copy.deepcopy(parameters),
        "source_inventory_contract": copy.deepcopy(parent["source_inventory_contract"]),
        "model_input_contract": copy.deepcopy(parent.get("model_input_contract")),
        "scene_disjoint": {
            "verified": not scene_overlap,
            "overlap": scene_overlap,
        },
        "artifacts": {
            "train_selection": {
                "file": "train_selection.jsonl",
                "records": len(selected_by_split["train"]),
                "sha256": hashlib.sha256(train_bytes).hexdigest(),
            },
            "val_selection": {
                "file": "val_selection.jsonl",
                "records": len(selected_by_split["val"]),
                "sha256": hashlib.sha256(val_bytes).hexdigest(),
            },
        },
        "ready": current_ready,
        "readiness_reason": readiness_reason,
        "splits": split_outputs,
    }
    output["manifest_sha256"] = canonical_sha256(output)
    manifest_bytes = json.dumps(
        output,
        indent=2,
        ensure_ascii=False,
        allow_nan=False,
    ).encode("utf-8")
    _publish_bundle_atomic(
        Path(output_dir),
        manifest_bytes=manifest_bytes,
        train_bytes=train_bytes,
        val_bytes=val_bytes,
    )
    return output


def main() -> int:
    args = parse_args()
    manifest = derive_multi_history_subset(
        args.parent_manifest,
        args.output_dir,
        train_samples=args.train_samples,
        val_samples=args.val_samples,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "ready": manifest["ready"],
                "train_samples": manifest["splits"]["train"]["selection_manifest"]["sample_count"],
                "val_samples": manifest["splits"]["val"]["selection_manifest"]["sample_count"],
                "manifest_sha256": manifest["manifest_sha256"],
            },
            sort_keys=True,
        )
    )
    return 0 if manifest["ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

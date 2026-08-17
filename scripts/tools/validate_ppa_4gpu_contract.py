#!/usr/bin/env python3
"""Fail-closed input/stage-transition checks for the formal PPA launcher.

The checker deliberately computes no file digest and acquires no file lock.
It validates identities and checkpoint semantics instead.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import sys
from pathlib import Path
from typing import Any, Mapping


READY_SCHEMA = "heatmapvln-amb3r-endpoint-pose-cache-ready-v2"
CACHE_SCHEMA = "heatmapvln-amb3r-causal-endpoint-training-cache-v2"
POSE_CONVENTION = "forward_m,left_m,cos_relative_yaw,sin_relative_yaw"
HISTORY_POSE_CONVENTION = (
    "habitat_c2w_minus_z__forward_left_cos_yaw_sin_yaw__v1"
)
ROW_POLICY = "official_map_update_endpoints_plus_final"
SNAPSHOT_TIMING = (
    "immediately_after_endpoint_mapping_before_ingesting_later_rgb"
)
CACHE_FILE = "amb3r_pose_cache.npz"
CACHE_MANIFEST = f"{CACHE_FILE}.json"

HEAD_PREFIXES = (
    "heatmap_vln.vit_dpt_fusion.",
    "heatmap_vln.vit_panorama_conditioner.",
    "heatmap_vln.coarse_panorama_conditioner.",
    "heatmap_vln.coarse.",
    "heatmap_vln.fine.",
)
FUTURE_PREFIX = "past_plan_action.future_head."
BRIDGE_PREFIX = "past_plan_action.bridge."
EXPECTED_HEAD_TENSORS = 79


class ContractError(RuntimeError):
    pass


def _resolved_dir(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_dir():
        raise ContractError(f"{label} is not a directory: {candidate}")
    return candidate.resolve()


def _resolved_file(path: str | Path, label: str) -> Path:
    candidate = Path(path).expanduser()
    if not candidate.is_file() or candidate.stat().st_size <= 0:
        raise ContractError(f"{label} is missing or empty: {candidate}")
    return candidate.resolve()


def _read_json(path: Path, label: str) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ContractError(f"invalid {label}: {path}: {exc}") from exc
    if not isinstance(value, dict):
        raise ContractError(f"{label} must be a JSON object: {path}")
    return value


def _expect(actual: Any, expected: Any, label: str) -> None:
    if actual != expected:
        raise ContractError(
            f"{label} mismatch: observed={actual!r}, expected={expected!r}"
        )


def _is_complete_dataset_clip(clip_dir: Path) -> bool:
    """Return False only for an abandoned, entirely empty clip placeholder."""
    meta_path = clip_dir / "meta.json"
    chunks = tuple((clip_dir / "chunks").glob("chunk_*.npz"))
    if meta_path.is_file() and chunks:
        return True
    if not meta_path.is_file() and not chunks:
        return False
    raise ContractError(
        "partially populated dataset clip is corrupt; expected both meta.json "
        f"and chunks/chunk_*.npz: {clip_dir}"
    )


def _discover_dataset_clip_keys(
    dataset_root: Path,
    required_splits: tuple[str, ...],
) -> tuple[str, ...]:
    """Reproduce the dataset's flat-scene MD5 train/val auto-split.

    The formal R2R root is ``r2r_paronamic_data/train`` and directly contains
    ``<scene>/clip_*``.  ``VLNSlidingWindowDataset`` hashes the sorted scene
    names when neither ``root/train`` nor ``root/val`` exists.  The cache gate
    intentionally supports only that formal layout; accepting a parent with
    explicit split directories would validate a different corpus identity.
    """
    if required_splits != ("train", "val"):
        raise ContractError(
            "formal PPA cache validation requires logical splits exactly "
            f"('train', 'val'), got {required_splits!r}"
        )

    # Reject (rather than silently reinterpret) <root>/<split>/... whenever a
    # logical split directory exists.  Its mere existence changes
    # VLNSlidingWindowDataset from auto-split to explicit-split mode for that
    # logical split, even when it happens to be empty.
    explicit_layouts: list[str] = []
    for split in required_splits:
        split_dir = dataset_root / split
        if split_dir.exists() or split_dir.is_symlink():
            explicit_layouts.append(str(split_dir))
    if explicit_layouts:
        raise ContractError(
            "explicit <root>/<train|val>/<scene>/clip_* layout is not supported "
            "by the formal PPA contract; pass the direct R2R scene root "
            f"r2r_paronamic_data/train instead: {explicit_layouts}"
        )

    scenes: list[tuple[Path, tuple[Path, ...]]] = []
    for scene_dir in sorted(dataset_root.iterdir(), key=lambda item: item.name):
        if scene_dir.is_symlink():
            raise ContractError(
                "dataset-root child may not be a symlink in the formal flat "
                f"scene layout: {scene_dir}"
            )
        if not scene_dir.is_dir():
            continue
        candidate_clip_dirs = tuple(
            sorted(
                (
                    child
                    for child in scene_dir.iterdir()
                    if child.is_dir() and child.name.startswith("clip_")
                ),
                key=lambda item: item.name,
            )
        )
        clip_dirs = tuple(
            clip_dir
            for clip_dir in candidate_clip_dirs
            if _is_complete_dataset_clip(clip_dir)
        )
        # Ignore annotations, control directories, and any other non-scene
        # directory exactly because they contain no direct clip_* children.
        if not clip_dirs:
            continue
        for clip_dir in clip_dirs:
            if clip_dir.is_symlink():
                raise ContractError(f"dataset clip may not be a symlink: {clip_dir}")
        scenes.append((scene_dir, clip_dirs))

    if not scenes:
        raise ContractError(
            f"no flat <scene>/clip_* directories found under {dataset_root}"
        )

    train_scenes: list[tuple[Path, tuple[Path, ...]]] = []
    val_scenes: list[tuple[Path, tuple[Path, ...]]] = []
    for scene in scenes:
        bucket = int(hashlib.md5(scene[0].name.encode()).hexdigest(), 16) % 100
        if bucket < 10:
            val_scenes.append(scene)
        else:
            train_scenes.append(scene)
    # This is the dataset's exact small-corpus fallback: because scenes were
    # name-sorted above, pop() moves the lexicographically final train scene.
    if not val_scenes and len(train_scenes) > 1:
        val_scenes.append(train_scenes.pop())

    scenes_by_split = {"train": train_scenes, "val": val_scenes}
    keys: list[str] = []
    for split in required_splits:
        selected = scenes_by_split[split]
        if not selected:
            raise ContractError(
                f"MD5 auto-split produced no {split} scenes under {dataset_root}"
            )
        for scene_dir, clip_dirs in selected:
            keys.extend(f"{scene_dir.name}/{clip_dir.name}" for clip_dir in clip_dirs)
    return tuple(keys)


def validate_cache(
    cache_root_raw: str,
    dataset_root_raw: str,
    required_splits: tuple[str, ...],
) -> dict[str, Any]:
    cache_root = _resolved_dir(cache_root_raw, "AMB3R cache root")
    dataset_root = _resolved_dir(dataset_root_raw, "expert scene root")
    ready_candidate = cache_root / "_control" / "cache.ready.json"
    if ready_candidate.is_symlink():
        raise ContractError(f"endpoint-v2 ready marker may not be a symlink: {ready_candidate}")
    ready_path = _resolved_file(
        ready_candidate,
        "endpoint-v2 ready marker",
    )
    ready = _read_json(ready_path, "endpoint-v2 ready marker")

    exact = {
        "schema": READY_SCHEMA,
        "complete": True,
        "causal": True,
        "endpoint_only": True,
        "failures": 0,
        "num_history": 8,
        "min_history": 5,
        "pose_convention": POSE_CONVENTION,
        "history_pose_convention": HISTORY_POSE_CONVENTION,
        "pose_provider": "amb3r_vo_da3",
        "translation_scale": 1.0,
        "query_only_at_map_endpoints": True,
        "query_every_frame_from_min_history": False,
        "query_every_frame": False,
        "row_policy": ROW_POLICY,
        "snapshot_timing": SNAPSHOT_TIMING,
        "future_pose_revisions_used": False,
    }
    for key, expected in exact.items():
        _expect(ready.get(key), expected, f"cache.ready.{key}")
    if "per_episode_gt_scale_used" in ready:
        _expect(
            ready.get("per_episode_gt_scale_used"),
            False,
            "cache.ready.per_episode_gt_scale_used",
        )
    if "gt_pose_read_by_exporter" in ready:
        _expect(
            ready.get("gt_pose_read_by_exporter"),
            False,
            "cache.ready.gt_pose_read_by_exporter",
        )

    _expect(
        _resolved_dir(ready.get("cache_root", ""), "ready cache_root"),
        cache_root,
        "cache.ready.cache_root",
    )
    _expect(
        _resolved_dir(ready.get("dataset_root", ""), "ready dataset_root"),
        dataset_root,
        "cache.ready.dataset_root",
    )
    observed_splits = ready.get("splits")
    if not isinstance(observed_splits, list):
        raise ContractError("cache.ready.splits must be a list")
    _expect(observed_splits, ["train", "val"], "cache.ready.splits")

    clip_keys = _discover_dataset_clip_keys(dataset_root, required_splits)
    _expect(ready.get("clips_total"), len(clip_keys), "cache.ready.clips_total")
    frames_total = ready.get("frames_total")
    rows_total = ready.get("query_rows_total")
    if isinstance(frames_total, bool) or not isinstance(frames_total, int) or frames_total < 1:
        raise ContractError("cache.ready.frames_total must be a positive integer")
    if isinstance(rows_total, bool) or not isinstance(rows_total, int) or rows_total < 1:
        raise ContractError("cache.ready.query_rows_total must be a positive integer")

    missing: list[str] = []
    for key in clip_keys:
        clip_cache = cache_root / key
        if clip_cache.is_symlink():
            raise ContractError(f"cache clip directory may not be a symlink: {clip_cache}")
        for filename in (CACHE_FILE, CACHE_MANIFEST):
            candidate = clip_cache / filename
            if (
                candidate.is_symlink()
                or not candidate.is_file()
                or candidate.stat().st_size <= 0
            ):
                missing.append(str(candidate))
                if len(missing) >= 8:
                    break
        if len(missing) >= 8:
            break
    if missing:
        raise ContractError(
            "endpoint-v2 ready marker claims full coverage but sidecars are "
            f"missing/empty: {missing}"
        )

    manifest_exact = {
        "schema": CACHE_SCHEMA,
        "causal": True,
        "num_history": 8,
        "min_history": 5,
        "pose_convention": POSE_CONVENTION,
        "history_pose_convention": HISTORY_POSE_CONVENTION,
        "pose_provider": "amb3r_vo_da3",
        "per_episode_gt_scale_used": False,
        "gt_pose_read_by_exporter": False,
        "endpoint_only": True,
        "row_policy": ROW_POLICY,
        "query_only_at_map_endpoints": True,
        "query_every_frame_from_min_history": False,
        "query_every_frame": False,
        "snapshot_timing": SNAPSHOT_TIMING,
        "future_pose_revisions_used": False,
        "translation_scale": 1.0,
    }
    manifest_frames_total = 0
    manifest_rows_total = 0
    for key in clip_keys:
        manifest_path = cache_root / key / CACHE_MANIFEST
        manifest = _read_json(manifest_path, "clip cache manifest")
        _expect(manifest.get("clip_key"), key, f"{key}.clip_key")
        for field, expected in manifest_exact.items():
            _expect(manifest.get(field), expected, f"{key}.{field}")
        for field in ("frame_count", "query_rows", "map_init_window", "map_every"):
            value = manifest.get(field)
            if isinstance(value, bool) or not isinstance(value, int) or value < 1:
                raise ContractError(f"{key}.{field} must be a positive integer")
        _expect(
            manifest.get("map_init_window"),
            ready.get("map_init_window"),
            f"{key}.map_init_window",
        )
        _expect(
            manifest.get("map_every"),
            ready.get("map_every"),
            f"{key}.map_every",
        )
        manifest_frames_total += int(manifest["frame_count"])
        manifest_rows_total += int(manifest["query_rows"])
    _expect(manifest_frames_total, frames_total, "ready/manifests frames_total")
    _expect(manifest_rows_total, rows_total, "ready/manifests query_rows_total")

    return {
        "status": "passed",
        "schema": READY_SCHEMA,
        "cache_root": str(cache_root),
        "dataset_root": str(dataset_root),
        "splits": list(required_splits),
        "clips": len(clip_keys),
        "query_rows": rows_total,
        "gt_pose_fallback_allowed": False,
        "checkpoint_hash_locking": False,
        "file_locking": False,
    }


def _normalize_key(name: str) -> str:
    if name.startswith("module."):
        name = name[len("module.") :]
    return name.replace(".module.", ".")


def _load_checkpoint(path: Path) -> Mapping[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ContractError("checkpoint validation requires PyTorch") from exc
    try:
        try:
            payload = torch.load(
                str(path), map_location="cpu", weights_only=True, mmap=True
            )
        except TypeError:
            payload = torch.load(str(path), map_location="cpu")
    except Exception as exc:
        raise ContractError(f"unable to load checkpoint {path}: {exc}") from exc
    if not isinstance(payload, Mapping):
        raise ContractError(f"checkpoint must contain a mapping: {path}")
    return payload


def _normalized_tensor_state(payload: Mapping[str, Any]) -> dict[str, Any]:
    try:
        import torch
    except ImportError as exc:
        raise ContractError("checkpoint validation requires PyTorch") from exc
    raw = payload.get("trainable_state_dict")
    if not isinstance(raw, Mapping) or not raw:
        raise ContractError(
            "checkpoint lacks non-empty deployment trainable_state_dict"
        )
    state: dict[str, Any] = {}
    for raw_name, value in raw.items():
        name = _normalize_key(str(raw_name))
        if name in state:
            raise ContractError(f"duplicate normalized checkpoint key: {name}")
        if not torch.is_tensor(value):
            raise ContractError(f"checkpoint state value is not a tensor: {name}")
        state[name] = value
    return state


def validate_checkpoint(path_raw: str, kind: str) -> dict[str, Any]:
    path = _resolved_file(path_raw, f"{kind} checkpoint")
    payload = _load_checkpoint(path)
    state = _normalized_tensor_state(payload)
    head = {name for name in state if name.startswith(HEAD_PREFIXES)}
    future = {name for name in state if name.startswith(FUTURE_PREFIX)}
    bridge = {name for name in state if name.startswith(BRIDGE_PREFIX)}
    _expect(len(head), EXPECTED_HEAD_TENSORS, "complete Heatmap Head tensor count")

    if kind == "past-init":
        unexpected = set(state) - head
        if unexpected:
            raise ContractError(
                "Stage-1 Past initializer must contain exactly the complete "
                f"79-tensor Head; unexpected={sorted(unexpected)[:8]}"
            )
        if future or bridge or payload.get("past_plan_action_contract") is not None:
            raise ContractError("Past initializer must predate PPA Stage 1")
        semantics = payload.get("weight_semantics")
        if semantics is not None and (
            not isinstance(semantics, Mapping)
            or "trainable_state_dict" not in semantics
        ):
            raise ContractError("invalid optional Past initializer weight_semantics")
        return {
            "status": "passed",
            "kind": kind,
            "path": str(path),
            "heatmap_head_tensors": len(head),
            "fresh_optimizer_scheduler_required": True,
            "checkpoint_digest_enforced": False,
            "file_lock_used": False,
        }

    manifest = payload.get("deployment_state_manifest")
    if not isinstance(manifest, Mapping):
        raise ContractError("checkpoint lacks deployment_state_manifest")
    _expect(
        manifest.get("self_contained_heatmap_head"),
        True,
        "deployment self_contained_heatmap_head",
    )
    _expect(
        manifest.get("heatmap_head_tensor_count"),
        EXPECTED_HEAD_TENSORS,
        "deployment heatmap_head_tensor_count",
    )
    semantics = payload.get("weight_semantics")
    if not isinstance(semantics, Mapping):
        raise ContractError("checkpoint lacks deployment weight_semantics")
    if "trainable_state_dict" not in semantics:
        raise ContractError("weight_semantics lacks trainable_state_dict entry")

    contract = payload.get("past_plan_action_contract")
    if not isinstance(contract, Mapping):
        raise ContractError("PPA checkpoint lacks past_plan_action_contract")
    expected_stage = {
        "stage1": "stage1_map_pretrain",
        "stage2": "stage2_joint",
    }[kind]
    _expect(contract.get("schema"), "past-plan-action-checkpoint-v1", "PPA schema")
    _expect(contract.get("stage"), expected_stage, "PPA stage")
    _expect(
        contract.get("complete_heatmap_head_tensors"),
        EXPECTED_HEAD_TENSORS,
        "PPA complete Heatmap Head tensors",
    )
    if not isinstance(contract.get("complete_future_head_tensors"), int):
        raise ContractError("PPA complete_future_head_tensors must be an integer")
    if contract["complete_future_head_tensors"] < 1:
        raise ContractError("PPA Future Head must contain at least one tensor")
    _expect(
        len(future),
        contract["complete_future_head_tensors"],
        "complete Future Head tensor count",
    )
    _expect(
        contract.get("stage1_to_stage2_fresh_optimizer"),
        True,
        "fresh Stage1->Stage2 optimizer contract",
    )
    _expect(
        contract.get("checkpoint_digest_enforced"),
        False,
        "checkpoint digest policy",
    )
    _expect(contract.get("file_lock_used"), False, "checkpoint file-lock policy")

    expected_bridge = kind == "stage2"
    _expect(
        contract.get("bridge_in_deployment_state"),
        expected_bridge,
        "bridge deployment-state policy",
    )
    if expected_bridge and not bridge:
        raise ContractError("Stage-2 deployment checkpoint lacks bridge tensors")
    if not expected_bridge and bridge:
        raise ContractError("Stage-1 deployment checkpoint must not contain bridge tensors")
    allowed = head | future | bridge
    unexpected = set(state) - allowed
    if unexpected:
        raise ContractError(
            f"PPA deployment entry contains unexpected tensors: {sorted(unexpected)[:8]}"
        )
    if payload.get("ema_state_dict") is None:
        raise ContractError("PPA stage transition requires an EMA deployment checkpoint")

    return {
        "status": "passed",
        "kind": kind,
        "path": str(path),
        "stage": expected_stage,
        "heatmap_head_tensors": len(head),
        "future_head_tensors": len(future),
        "bridge_tensors": len(bridge),
        "fresh_optimizer_scheduler_required": True,
        "checkpoint_digest_enforced": False,
        "file_lock_used": False,
    }


def validate_config(schema_raw: str, config_raw: str, expected_stage: str) -> dict[str, Any]:
    schema_path = _resolved_file(schema_raw, "config schema")
    config_path = _resolved_file(config_raw, "training config")
    spec = importlib.util.spec_from_file_location("ppa_live_config_schema", schema_path)
    if spec is None or spec.loader is None:
        raise ContractError(f"cannot import config schema: {schema_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    cfg = module.load_and_validate_config(config_path)
    stages = cfg["training"]["stages"]
    if len(stages) != 1:
        raise ContractError("each PPA invocation must contain exactly one stage")
    stage = stages[0]
    _expect(stage.get("past_plan_action_stage"), expected_stage, "configured PPA stage")
    _expect(cfg["data"]["dataset_type"], "trajectory", "expert-only dataset type")
    trajectory = cfg["data"]["trajectory"]
    _expect(trajectory["require_amb3r_pose_cache"], True, "strict AMB3R cache")
    _expect(trajectory["random_subsequence"], False, "endpoint cache sampling")
    _expect(trajectory["max_clips"], 0, "formal max_clips")
    _expect(stage["required_history_pose_provider"], "amb3r_vo_cache", "pose provider")
    _expect(stage["trainable_modules"], ["past_plan_action", "heatmap_vln"], "trainable modules")
    _expect(stage["train_action"], expected_stage == "stage2_joint", "action training")
    effective_batch = (
        int(cfg["optim"]["batch_size"])
        * 4
        * int(cfg["optim"]["grad_accum_steps"])
    )
    _expect(effective_batch, 8, "effective global batch")
    unresolved = []

    def walk(value: Any, path: str) -> None:
        if isinstance(value, str) and "$" in value:
            unresolved.append(path)
        elif isinstance(value, Mapping):
            for key, child in value.items():
                walk(child, f"{path}.{key}")
        elif isinstance(value, list):
            for index, child in enumerate(value):
                walk(child, f"{path}[{index}]")

    walk(cfg, "config")
    if unresolved:
        raise ContractError(f"unexpanded environment variables: {unresolved[:8]}")
    return {
        "status": "passed",
        "config": str(config_path),
        "stage": expected_stage,
        "world_size": 4,
        "per_rank_batch": cfg["optim"]["batch_size"],
        "gradient_accumulation": cfg["optim"]["grad_accum_steps"],
        "effective_global_batch": effective_batch,
    }


def resolve_completed_best(output_root_raw: str, kind: str) -> Path:
    output_root = _resolved_dir(output_root_raw, f"{kind} output root")
    latest_link = output_root / "latest"
    if not latest_link.is_symlink():
        raise ContractError(f"completed run must expose latest symlink: {latest_link}")
    run_dir = latest_link.resolve(strict=True)
    try:
        run_dir.relative_to(output_root)
    except ValueError as exc:
        raise ContractError(f"latest escapes its stage output root: {run_dir}") from exc
    summary = _read_json(
        _resolved_file(run_dir / "manifest" / "summary.json", "run summary"),
        "run summary",
    )
    _expect(
        Path(str(summary.get("run_dir", ""))).resolve(),
        run_dir,
        "summary run_dir",
    )
    metrics_path = _resolved_file(run_dir / "logs" / "metrics.jsonl", "metrics log")
    last_line = ""
    with metrics_path.open(encoding="utf-8") as handle:
        for line in handle:
            if line.strip():
                last_line = line
    if not last_line:
        raise ContractError("metrics log is empty")
    try:
        final_record = json.loads(last_line)
    except json.JSONDecodeError as exc:
        raise ContractError("metrics log ends with partial JSON") from exc
    _expect(final_record.get("record_type"), "run_complete", "final metrics record")
    best = _resolved_file(run_dir / "checkpoints" / "best.pth", f"{kind} best")
    validate_checkpoint(str(best), kind)
    return best


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    cache = subparsers.add_parser("cache")
    cache.add_argument("--cache-root", required=True)
    cache.add_argument("--dataset-root", required=True)
    cache.add_argument("--required-splits", nargs="+", default=["train", "val"])

    checkpoint = subparsers.add_parser("checkpoint")
    checkpoint.add_argument("--path", required=True)
    checkpoint.add_argument("--kind", choices=["past-init", "stage1", "stage2"], required=True)

    config = subparsers.add_parser("config")
    config.add_argument("--schema", required=True)
    config.add_argument("--config", required=True)
    config.add_argument(
        "--expected-stage",
        choices=["stage1_map_pretrain", "stage2_joint"],
        required=True,
    )

    run_best = subparsers.add_parser("run-best")
    run_best.add_argument("--output-root", required=True)
    run_best.add_argument("--kind", choices=["stage1", "stage2"], required=True)
    return parser


def main() -> int:
    args = _parser().parse_args()
    try:
        if args.command == "cache":
            result = validate_cache(
                args.cache_root,
                args.dataset_root,
                tuple(args.required_splits),
            )
            print(json.dumps(result, sort_keys=True))
        elif args.command == "checkpoint":
            print(json.dumps(validate_checkpoint(args.path, args.kind), sort_keys=True))
        elif args.command == "config":
            print(
                json.dumps(
                    validate_config(
                        args.schema,
                        args.config,
                        args.expected_stage,
                    ),
                    sort_keys=True,
                )
            )
        elif args.command == "run-best":
            print(resolve_completed_best(args.output_root, args.kind))
        else:  # pragma: no cover
            raise AssertionError(args.command)
    except ContractError as exc:
        print(f"PPA contract error: {exc}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

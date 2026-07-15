#!/usr/bin/env python3
"""Validate the paired Stage2 contract for the Task39 downstream experiment.

The validator is intentionally fail-closed.  It accepts only two completed
epoch-3 h1024 adapter checkpoints that share every training argument except
the treatment checkpoint and output directory, were resumed from the same
explicit epoch-0 adapter initialization, and used byte-identical data splits.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.tools.export_pose_free_lora_wrapper import (
    file_sha256,
    tensor_state_sha256,
)

SUMMARY_SCHEMA = "task39_stage2_pair_validation_v1"
EXPECTED_EPOCH = 3
EXPECTED_ADAPTER_TYPE = "pano_latent_space"
EXPECTED_ADAPTER_HIDDEN_DIM = 1024
EXPECTED_ADAPTER_TENSORS = 4
EXPECTED_ADAPTER_NUMEL = 7_344_640
EXPECTED_SEED = 42
EXPECTED_EPOCHS = 3
EXPECTED_BATCH_SIZE = 16
EXPECTED_LR = 3.0e-4
EXPECTED_ADAPTER_LAYOUT: dict[str, tuple[tuple[int, ...], torch.dtype]] = {
    "mlp.0.weight": ((1024, 3584), torch.float32),
    "mlp.0.bias": ((1024,), torch.float32),
    "mlp.3.weight": ((3584, 1024), torch.float32),
    "mlp.3.bias": ((3584,), torch.float32),
}
EXPECTED_FIXED_ARGS: dict[str, Any] = {
    "seed": EXPECTED_SEED,
    "epochs": EXPECTED_EPOCHS,
    "batch_size": EXPECTED_BATCH_SIZE,
    "lr": EXPECTED_LR,
    "weight_decay": 0.01,
    "grad_clip": 1.0,
    "adapter_hidden_dim": EXPECTED_ADAPTER_HIDDEN_DIM,
    "adapter_dropout": 0.0,
    "max_samples": 0,
    "dataset_max_clips": 0,
    "val_ratio": 0.1,
    "val_records": 0,
    "index_mode": "generic",
    "split": "train",
    "pano_max_side_dist_m": 6.0,
    "prefetch_batches": 8,
    "prefetch_workers": 4,
    "startup_preflight_batches": 4,
    "teacher_target_mode": "native_sidecar",
    "compute_teacher_mse": False,
    "teacher_torch_dtype": "bfloat16",
    "teacher_attn_implementation": "sdpa",
    "teacher_flash_attn_stub": True,
    "teacher_cache_mode": "unbounded",
    "teacher_cache_max_items": 0,
    "teacher_preload_cache": True,
    "teacher_preload_workers": 8,
    "check_teacher_tensor_files": False,
    "trust_native_sidecar_pano_labels": True,
    "raw_distill_weight": 0.1,
    "raw_norm_weight": 0.1,
    "cond_distill_weight": 1.0,
    "cond_cosine_weight": 1.0,
    "cond_smooth_l1_beta": 1.0,
    "gt_weight": 0.2,
    "save_every_epochs": 1,
    "ddp_backend": "auto",
}
ARM_SPECIFIC_ARG_KEYS = frozenset({"base_checkpoint", "output_dir"})


class Stage2PairValidationError(RuntimeError):
    """Raised when a Stage2 arm or the paired fairness contract is invalid."""


def _resolve_input_file(path: str | Path, *, label: str) -> Path:
    try:
        resolved = Path(path).expanduser().resolve(strict=True)
    except FileNotFoundError as exc:
        raise FileNotFoundError(f"{label} does not exist: {path}") from exc
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} is not a file: {resolved}")
    return resolved


def _prepare_output_path(path: str | Path) -> Path:
    raw = Path(path).expanduser()
    if raw.exists() or raw.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing file: {raw}")
    resolved = raw.resolve()
    if resolved.exists() or resolved.is_symlink():
        raise FileExistsError(f"Refusing to overwrite existing file: {resolved}")
    return resolved


def _require_mapping(payload: Mapping[str, Any], key: str, *, arm: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise Stage2PairValidationError(f"{arm} checkpoint is missing mapping: {key}")
    return dict(value)


def _require_path_arg(args: Mapping[str, Any], key: str, *, arm: str) -> Path:
    value = args.get(key)
    if not isinstance(value, (str, os.PathLike)) or not str(value):
        raise Stage2PairValidationError(f"{arm} args.{key} must be a non-empty path")
    return _resolve_input_file(value, label=f"{arm} args.{key}")


def _structural_value(value: Any) -> tuple[Any, ...]:
    """Return a type-sensitive, deterministic representation of an arg value."""

    if value is None:
        return ("none",)
    if isinstance(value, bool):
        return ("bool", value)
    if isinstance(value, int):
        return ("int", value)
    if isinstance(value, float):
        return ("float", value.hex())
    if isinstance(value, str):
        return ("str", value)
    if isinstance(value, os.PathLike):
        return ("path", os.fspath(value))
    if isinstance(value, list):
        return ("list", tuple(_structural_value(item) for item in value))
    if isinstance(value, tuple):
        return ("tuple", tuple(_structural_value(item) for item in value))
    if isinstance(value, Mapping):
        items: list[tuple[str, tuple[Any, ...]]] = []
        for key, item in value.items():
            if not isinstance(key, str):
                raise Stage2PairValidationError(f"Stage2 args contains unsupported non-string mapping key: {key!r}")
            items.append((key, _structural_value(item)))
        return ("mapping", tuple(sorted(items)))
    raise Stage2PairValidationError(
        f"Stage2 args contains unsupported value type {type(value).__module__}.{type(value).__qualname__}"
    )


def _arg_differences(
    warm_args: Mapping[str, Any],
    control_args: Mapping[str, Any],
) -> list[str]:
    differences: list[str] = []
    keys = sorted((set(warm_args) | set(control_args)) - ARM_SPECIFIC_ARG_KEYS)
    for key in keys:
        if key not in warm_args:
            differences.append(f"{key} (missing from warm)")
        elif key not in control_args:
            differences.append(f"{key} (missing from control)")
        elif _structural_value(warm_args[key]) != _structural_value(control_args[key]):
            differences.append(key)
    return differences


def _require_fixed_training_args(args: Mapping[str, Any], *, arm: str) -> None:
    for key, expected in EXPECTED_FIXED_ARGS.items():
        if key not in args:
            raise Stage2PairValidationError(f"{arm} args.{key} is missing, expected exact value {expected!r}")
        actual = args[key]
        if _structural_value(actual) != _structural_value(expected):
            raise Stage2PairValidationError(f"{arm} args.{key}={actual!r}, expected exact value {expected!r}")


def _load_and_validate_checkpoint(path: Path, *, arm: str) -> dict[str, Any]:
    file_sha_before = file_sha256(path)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except Exception as exc:
        raise Stage2PairValidationError(f"Could not load {arm} checkpoint {path}: {exc}") from exc
    file_sha_after = file_sha256(path)
    if file_sha_after != file_sha_before:
        raise Stage2PairValidationError(f"{arm} checkpoint changed while being read: {path}")
    if not isinstance(payload, Mapping):
        raise Stage2PairValidationError(f"{arm} checkpoint payload is not a mapping")

    epoch = payload.get("epoch")
    if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch != EXPECTED_EPOCH:
        raise Stage2PairValidationError(f"{arm} checkpoint epoch={epoch!r}, expected {EXPECTED_EPOCH}")
    step = payload.get("step")
    if isinstance(step, bool) or not isinstance(step, int) or step <= 0:
        raise Stage2PairValidationError(f"{arm} checkpoint step must be a positive integer, got {step!r}")
    adapter_type = payload.get("adapter_type")
    if adapter_type != EXPECTED_ADAPTER_TYPE:
        raise Stage2PairValidationError(
            f"{arm} checkpoint adapter_type={adapter_type!r}, expected {EXPECTED_ADAPTER_TYPE!r}"
        )

    args = _require_mapping(payload, "args", arm=arm)

    raw_state = payload.get("adapter_state_dict")
    if not isinstance(raw_state, Mapping):
        raise Stage2PairValidationError(f"{arm} checkpoint is missing mapping: adapter_state_dict")
    state: dict[str, torch.Tensor] = {}
    for name, tensor in raw_state.items():
        if not isinstance(name, str) or not name:
            raise Stage2PairValidationError(f"{arm} adapter_state_dict contains a non-string or empty tensor name")
        if not torch.is_tensor(tensor):
            raise Stage2PairValidationError(f"{arm} adapter_state_dict[{name!r}] is not a tensor")
        state[name] = tensor

    if len(state) != EXPECTED_ADAPTER_TENSORS:
        raise Stage2PairValidationError(f"{arm} adapter tensor count={len(state)}, expected {EXPECTED_ADAPTER_TENSORS}")
    expected_names = set(EXPECTED_ADAPTER_LAYOUT)
    actual_names = set(state)
    if actual_names != expected_names:
        missing = sorted(expected_names - actual_names)
        unexpected = sorted(actual_names - expected_names)
        raise Stage2PairValidationError(
            f"{arm} adapter tensor names mismatch: missing={missing} unexpected={unexpected}"
        )
    adapter_numel = sum(tensor.numel() for tensor in state.values())
    if adapter_numel != EXPECTED_ADAPTER_NUMEL:
        raise Stage2PairValidationError(f"{arm} adapter numel={adapter_numel}, expected {EXPECTED_ADAPTER_NUMEL}")
    for name, (expected_shape, expected_dtype) in EXPECTED_ADAPTER_LAYOUT.items():
        tensor = state[name]
        if tensor.layout != torch.strided:
            raise Stage2PairValidationError(
                f"{arm} adapter tensor {name!r} layout={tensor.layout}, expected torch.strided"
            )
        if tuple(tensor.shape) != expected_shape:
            raise Stage2PairValidationError(
                f"{arm} adapter tensor {name!r} shape={tuple(tensor.shape)}, expected {expected_shape}"
            )
        if tensor.dtype != expected_dtype:
            raise Stage2PairValidationError(
                f"{arm} adapter tensor {name!r} dtype={tensor.dtype}, expected {expected_dtype}"
            )
    nonfinite = [name for name, tensor in state.items() if not bool(torch.isfinite(tensor.detach()).all().item())]
    if nonfinite:
        raise Stage2PairValidationError(f"{arm} adapter contains non-finite tensors: {nonfinite}")

    return {
        "path": path,
        "file_sha256": file_sha_before,
        "adapter_state_sha256": tensor_state_sha256(state),
        "adapter_tensor_count": len(state),
        "adapter_numel": adapter_numel,
        "epoch": epoch,
        "step": step,
        "args": args,
    }


def _read_split(path: Path, *, arm: str) -> tuple[bytes, str]:
    try:
        payload = path.read_bytes()
    except OSError as exc:
        raise Stage2PairValidationError(f"Could not read {arm} split file {path}: {exc}") from exc
    return payload, hashlib.sha256(payload).hexdigest()


def _write_json_no_replace(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.",
        suffix=".tmp",
        dir=path.parent,
    )
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError as exc:
            raise FileExistsError(f"Refusing to overwrite existing file: {path}") from exc
    finally:
        temporary.unlink(missing_ok=True)


def validate_task39_stage2_pair(
    warm_checkpoint: str | Path,
    control_checkpoint: str | Path,
    warm_split: str | Path,
    control_split: str | Path,
    expected_warm_base: str | Path,
    expected_control_base: str | Path,
    shared_init: str | Path,
    *,
    output_json: str | Path | None = None,
) -> dict[str, Any]:
    """Validate two completed Stage2 arms and optionally publish a JSON report."""

    output_path = _prepare_output_path(output_json) if output_json is not None else None

    warm_checkpoint_path = _resolve_input_file(warm_checkpoint, label="warm checkpoint")
    control_checkpoint_path = _resolve_input_file(control_checkpoint, label="control checkpoint")
    warm_split_path = _resolve_input_file(warm_split, label="warm split")
    control_split_path = _resolve_input_file(control_split, label="control split")
    expected_warm_base_path = _resolve_input_file(expected_warm_base, label="expected warm base")
    expected_control_base_path = _resolve_input_file(
        expected_control_base,
        label="expected control base",
    )
    shared_init_path = _resolve_input_file(shared_init, label="shared init")

    warm = _load_and_validate_checkpoint(warm_checkpoint_path, arm="warm")
    control = _load_and_validate_checkpoint(control_checkpoint_path, arm="control")
    if warm["step"] != control["step"]:
        raise Stage2PairValidationError(
            f"Stage2 optimizer-step counts differ: warm={warm['step']} control={control['step']}"
        )
    warm_args = warm["args"]
    control_args = control["args"]

    for arm, args in (("warm", warm_args), ("control", control_args)):
        _require_fixed_training_args(args, arm=arm)
        output_dir = args.get("output_dir")
        if not isinstance(output_dir, (str, os.PathLike)) or not str(output_dir):
            raise Stage2PairValidationError(f"{arm} args.output_dir must be a non-empty path")

    recorded_warm_base = _require_path_arg(warm_args, "base_checkpoint", arm="warm")
    recorded_control_base = _require_path_arg(control_args, "base_checkpoint", arm="control")
    if recorded_warm_base != expected_warm_base_path:
        raise Stage2PairValidationError(
            f"warm args.base_checkpoint mismatch: recorded={recorded_warm_base} expected={expected_warm_base_path}"
        )
    if recorded_control_base != expected_control_base_path:
        raise Stage2PairValidationError(
            "control args.base_checkpoint mismatch: "
            f"recorded={recorded_control_base} expected={expected_control_base_path}"
        )

    warm_resume = _require_path_arg(warm_args, "resume_adapter", arm="warm")
    control_resume = _require_path_arg(control_args, "resume_adapter", arm="control")
    if warm_resume != shared_init_path or control_resume != shared_init_path:
        raise Stage2PairValidationError(
            "Both args.resume_adapter paths must resolve to the expected shared init: "
            f"warm={warm_resume} control={control_resume} expected={shared_init_path}"
        )

    warm_teacher = _require_path_arg(warm_args, "teacher_jsonl", arm="warm")
    control_teacher = _require_path_arg(control_args, "teacher_jsonl", arm="control")
    if warm_teacher != control_teacher:
        raise Stage2PairValidationError(
            f"Stage2 teacher sidecars differ: warm={warm_teacher} control={control_teacher}"
        )

    arg_differences = _arg_differences(warm_args, control_args)
    if arg_differences:
        raise Stage2PairValidationError(
            "Stage2 args differ outside base_checkpoint/output_dir: " + ", ".join(arg_differences)
        )

    warm_split_bytes, warm_split_sha = _read_split(warm_split_path, arm="warm")
    control_split_bytes, control_split_sha = _read_split(control_split_path, arm="control")
    if warm_split_bytes != control_split_bytes:
        raise Stage2PairValidationError(
            "Stage2 split files are not byte-identical: "
            f"warm_sha256={warm_split_sha} control_sha256={control_split_sha}"
        )

    summary: dict[str, Any] = {
        "schema": SUMMARY_SCHEMA,
        "contract": {
            "epoch": EXPECTED_EPOCH,
            "adapter_type": EXPECTED_ADAPTER_TYPE,
            "adapter_hidden_dim": EXPECTED_ADAPTER_HIDDEN_DIM,
            "adapter_tensor_count": EXPECTED_ADAPTER_TENSORS,
            "adapter_numel": EXPECTED_ADAPTER_NUMEL,
            "adapter_dtype": str(torch.float32),
            "fixed_training_args": dict(EXPECTED_FIXED_ARGS),
        },
        "arms": {
            "warm": {
                "checkpoint_path": str(warm_checkpoint_path),
                "checkpoint_file_sha256": warm["file_sha256"],
                "adapter_state_sha256": warm["adapter_state_sha256"],
                "adapter_tensor_count": warm["adapter_tensor_count"],
                "adapter_numel": warm["adapter_numel"],
                "epoch": warm["epoch"],
                "step": warm["step"],
                "split_path": str(warm_split_path),
                "split_sha256": warm_split_sha,
                "expected_base_path": str(expected_warm_base_path),
                "recorded_output_dir": os.fspath(warm_args["output_dir"]),
            },
            "control": {
                "checkpoint_path": str(control_checkpoint_path),
                "checkpoint_file_sha256": control["file_sha256"],
                "adapter_state_sha256": control["adapter_state_sha256"],
                "adapter_tensor_count": control["adapter_tensor_count"],
                "adapter_numel": control["adapter_numel"],
                "epoch": control["epoch"],
                "step": control["step"],
                "split_path": str(control_split_path),
                "split_sha256": control_split_sha,
                "expected_base_path": str(expected_control_base_path),
                "recorded_output_dir": os.fspath(control_args["output_dir"]),
            },
        },
        "shared_init_path": str(shared_init_path),
        "shared_init_file_sha256": file_sha256(shared_init_path),
        "teacher_sidecar_path": str(warm_teacher),
        "split_sha256": warm_split_sha,
        "args_comparison_excluded_keys": sorted(ARM_SPECIFIC_ARG_KEYS),
        "checks": {
            "warm_checkpoint_contract": True,
            "control_checkpoint_contract": True,
            "optimizer_step_count_identical": True,
            "expected_base_paths": True,
            "shared_init_resume": True,
            "teacher_sidecar_identical": True,
            "args_identical_except_arm_paths": True,
            "split_bytes_identical": True,
            "fixed_training_args": True,
            "adapter_layout_and_dtype": True,
        },
        "passed": True,
    }

    if output_path is not None:
        _write_json_no_replace(output_path, summary)
    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warm-checkpoint", type=Path, required=True)
    parser.add_argument("--control-checkpoint", type=Path, required=True)
    parser.add_argument("--warm-split", type=Path, required=True)
    parser.add_argument("--control-split", type=Path, required=True)
    parser.add_argument("--expected-warm-base", type=Path, required=True)
    parser.add_argument("--expected-control-base", type=Path, required=True)
    parser.add_argument("--shared-init", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, default=None)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = validate_task39_stage2_pair(
        args.warm_checkpoint,
        args.control_checkpoint,
        args.warm_split,
        args.control_split,
        args.expected_warm_base,
        args.expected_control_base,
        args.shared_init,
        output_json=args.output_json,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

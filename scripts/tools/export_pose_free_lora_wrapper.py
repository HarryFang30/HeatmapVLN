#!/usr/bin/env python3
"""Export a strict Task38 checkpoint as a Stage2/Stage3 LoRA-only wrapper.

The Task38 visual-identity pilot stores its learned adapter under the top-level
``lora_state_dict`` key, while the existing training loaders consume a
top-level ``trainable_state_dict``.  This tool changes only that container
shape.  Tensor names, dtypes, shapes, and bytes are preserved and verified
before the wrapper is published.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch

SOURCE_SCHEMA = "pose_free_visual_identity_checkpoint_v3"
SOURCE_PROTOCOL = "strict_b1_visual_identity_two_stage_v3"
SOURCE_TRAIN_MODES = ("head-warmup", "lora-identity", "lora-heatmap-control")
WRAPPER_SCHEMA = "stage2_stage3_lora_only_wrapper_v1"
WRAPPER_PROTOCOL = "task38_pose_free_lora_only_export_v1"
SUMMARY_SCHEMA = "task38_pose_free_lora_only_export_summary_v1"
DEFAULT_EXPECTED_LORA_TENSORS = 224


class CheckpointExportError(RuntimeError):
    """Raised when a checkpoint fails an export contract."""


def file_sha256(path: str | Path) -> str:
    """Return the SHA-256 digest of a file's exact bytes."""

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def tensor_state_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, dtypes, shapes, and bytes using the Task38 contract."""

    digest = hashlib.sha256(b"task36c_tensor_state_v1\0")
    for name, tensor in sorted(state.items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(b"\0")
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(b"\0")
        digest.update(json.dumps(list(value.shape), separators=(",", ":")).encode("ascii"))
        digest.update(b"\0")
        digest.update(value.reshape(-1).view(torch.uint8).numpy().tobytes(order="C"))
    return digest.hexdigest()


def _require_string(payload: Mapping[str, Any], key: str) -> str:
    value = payload.get(key)
    if not isinstance(value, str) or not value:
        raise CheckpointExportError(f"Source checkpoint is missing non-empty string metadata: {key}")
    return value


def _validate_sha256_pin(value: str | None, *, label: str) -> str | None:
    if value is None:
        return None
    if len(value) != 64 or any(character not in "0123456789abcdef" for character in value):
        raise ValueError(f"{label} must be exactly 64 lowercase hexadecimal characters")
    return value


def _require_tensor_state(
    payload: Mapping[str, Any],
    key: str,
    *,
    require_lora_names: bool = False,
) -> dict[str, torch.Tensor]:
    value = payload.get(key)
    if not isinstance(value, Mapping) or not value:
        raise CheckpointExportError(f"Source checkpoint is missing non-empty {key}")

    state: dict[str, torch.Tensor] = {}
    for name, tensor in value.items():
        if not isinstance(name, str) or not name:
            raise CheckpointExportError(f"{key} contains a non-string or empty tensor name")
        if not torch.is_tensor(tensor):
            raise CheckpointExportError(f"{key}[{name!r}] is not a tensor")
        if require_lora_names and "lora_" not in name:
            raise CheckpointExportError(f"{key}[{name!r}] is not a LoRA tensor name")
        state[name] = tensor
    return state


def _validate_source_payload(
    payload: Any,
    *,
    expected_lora_tensors: int | None,
    expected_lora_sha256: str | None,
) -> tuple[dict[str, torch.Tensor], str, str, str, int, str, str]:
    if not isinstance(payload, Mapping):
        raise CheckpointExportError("Source checkpoint payload is not a mapping")

    source_schema = _require_string(payload, "schema")
    source_protocol = _require_string(payload, "protocol")
    if source_schema != SOURCE_SCHEMA:
        raise CheckpointExportError(f"Source schema mismatch: expected={SOURCE_SCHEMA!r} actual={source_schema!r}")
    if source_protocol != SOURCE_PROTOCOL:
        raise CheckpointExportError(
            f"Source protocol mismatch: expected={SOURCE_PROTOCOL!r} actual={source_protocol!r}"
        )

    source_train_mode = _require_string(payload, "train_mode")
    if source_train_mode not in SOURCE_TRAIN_MODES:
        raise CheckpointExportError(f"Source checkpoint has unsupported train_mode: {source_train_mode!r}")
    source_step = payload.get("step")
    if isinstance(source_step, bool) or not isinstance(source_step, int) or source_step <= 0:
        raise CheckpointExportError("Source checkpoint has invalid positive step metadata")

    lora_state = _require_tensor_state(payload, "lora_state_dict", require_lora_names=True)
    head_state = _require_tensor_state(payload, "head_state_dict")
    actual_count = len(lora_state)

    metadata_count = payload.get("expected_lora_tensors")
    if isinstance(metadata_count, bool) or not isinstance(metadata_count, int) or metadata_count <= 0:
        raise CheckpointExportError("Source checkpoint has invalid expected_lora_tensors metadata")
    if metadata_count != actual_count:
        raise CheckpointExportError(
            f"Source LoRA count does not match its metadata: metadata={metadata_count} actual={actual_count}"
        )
    if expected_lora_tensors is not None:
        if expected_lora_tensors <= 0:
            raise ValueError("expected_lora_tensors must be positive or None")
        if actual_count != expected_lora_tensors:
            raise CheckpointExportError(
                f"LoRA tensor count mismatch: expected={expected_lora_tensors} actual={actual_count}"
            )

    metadata_lora_sha = _require_string(payload, "lora_state_sha256")
    actual_lora_sha = tensor_state_sha256(lora_state)
    if actual_lora_sha != metadata_lora_sha:
        raise CheckpointExportError(
            f"Source lora_state_dict strong hash mismatch: metadata={metadata_lora_sha} actual={actual_lora_sha}"
        )
    if expected_lora_sha256 is not None and actual_lora_sha != expected_lora_sha256:
        raise CheckpointExportError(
            f"Source LoRA SHA-256 pin mismatch: expected={expected_lora_sha256} actual={actual_lora_sha}"
        )

    metadata_head_sha = _require_string(payload, "head_state_sha256")
    actual_head_sha = tensor_state_sha256(head_state)
    if actual_head_sha != metadata_head_sha:
        raise CheckpointExportError(
            f"Source head_state_dict strong hash mismatch: metadata={metadata_head_sha} actual={actual_head_sha}"
        )
    return (
        lora_state,
        source_schema,
        source_protocol,
        source_train_mode,
        source_step,
        actual_lora_sha,
        actual_head_sha,
    )


def _clone_cpu_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {name: tensor.detach().cpu().clone(memory_format=torch.preserve_format) for name, tensor in state.items()}


def _temporary_path(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    file_descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    os.close(file_descriptor)
    return Path(temporary_name)


def _fsync_file(path: Path) -> None:
    with path.open("rb") as handle:
        os.fsync(handle.fileno())


def _publish_temporary(temporary: Path, destination: Path, *, overwrite: bool) -> None:
    """Publish a complete same-directory file atomically.

    The hard-link path supplies atomic no-replace semantics.  ``os.replace``
    is used only when the caller explicitly opts into overwriting.
    """

    if overwrite:
        os.replace(temporary, destination)
        return
    try:
        os.link(temporary, destination)
    except FileExistsError as exc:
        raise FileExistsError(f"Refusing to overwrite existing file: {destination}") from exc
    temporary.unlink()


def _write_json_temporary(path: Path, payload: Mapping[str, Any]) -> Path:
    temporary = _temporary_path(path)
    try:
        with temporary.open("w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2, ensure_ascii=False, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
    except Exception:
        temporary.unlink(missing_ok=True)
        raise
    return temporary


def _default_summary_path(output_path: Path) -> Path:
    return output_path.with_name(f"{output_path.name}.summary.json")


def export_pose_free_lora_wrapper(
    source_checkpoint: str | Path,
    output_path: str | Path,
    *,
    summary_json: str | Path | None = None,
    expected_lora_tensors: int | None = DEFAULT_EXPECTED_LORA_TENSORS,
    expected_source_sha256: str | None = None,
    expected_lora_sha256: str | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Validate and export one Task38 checkpoint.

    Returns the same JSON-safe summary that is written beside the wrapper.
    """

    expected_source_sha256 = _validate_sha256_pin(
        expected_source_sha256,
        label="expected_source_sha256",
    )
    expected_lora_sha256 = _validate_sha256_pin(
        expected_lora_sha256,
        label="expected_lora_sha256",
    )
    source = Path(source_checkpoint).expanduser().resolve(strict=True)
    output = Path(output_path).expanduser().resolve()
    summary_path = (
        Path(summary_json).expanduser().resolve() if summary_json is not None else _default_summary_path(output)
    )
    if not source.is_file():
        raise FileNotFoundError(f"Source checkpoint is not a file: {source}")
    if len({source, output, summary_path}) != 3:
        raise ValueError("Source, wrapper output, and JSON summary paths must be distinct")
    if not overwrite:
        for path in (output, summary_path):
            if path.exists():
                raise FileExistsError(f"Refusing to overwrite existing file: {path}")

    source_file_sha = file_sha256(source)
    if expected_source_sha256 is not None and source_file_sha != expected_source_sha256:
        raise CheckpointExportError(
            f"Source file SHA-256 pin mismatch: expected={expected_source_sha256} actual={source_file_sha}"
        )
    payload = torch.load(source, map_location="cpu", weights_only=True)
    source_file_sha_after_load = file_sha256(source)
    if source_file_sha_after_load != source_file_sha:
        raise CheckpointExportError("Source checkpoint changed while it was being read; export refused")

    (
        lora_state,
        source_schema,
        source_protocol,
        source_train_mode,
        source_step,
        lora_sha,
        head_sha,
    ) = _validate_source_payload(
        payload,
        expected_lora_tensors=expected_lora_tensors,
        expected_lora_sha256=expected_lora_sha256,
    )
    trainable_state = _clone_cpu_state(lora_state)
    if tensor_state_sha256(trainable_state) != lora_sha:
        raise CheckpointExportError("CPU clone changed the LoRA tensor-state hash")

    wrapper: dict[str, Any] = {
        "schema": WRAPPER_SCHEMA,
        "protocol": WRAPPER_PROTOCOL,
        "source_path": str(source),
        "source_file_sha256": source_file_sha,
        "source_schema": source_schema,
        "source_protocol": source_protocol,
        "source_train_mode": source_train_mode,
        "source_step": source_step,
        "source_lora_state_sha256": lora_sha,
        "source_head_state_sha256": head_sha,
        "expected_source_file_sha256": expected_source_sha256,
        "expected_lora_state_sha256": expected_lora_sha256,
        "lora_tensor_count": len(trainable_state),
        "expected_lora_tensors": expected_lora_tensors,
        "trainable_state_sha256": lora_sha,
        "trainable_state_dict": trainable_state,
    }

    output_temporary = _temporary_path(output)
    summary_temporary: Path | None = None
    output_published = False
    try:
        torch.save(wrapper, output_temporary)
        _fsync_file(output_temporary)

        roundtrip = torch.load(output_temporary, map_location="cpu", weights_only=True)
        if not isinstance(roundtrip, Mapping) or roundtrip.get("schema") != WRAPPER_SCHEMA:
            raise CheckpointExportError("Wrapper round-trip schema verification failed")
        roundtrip_state = _require_tensor_state(roundtrip, "trainable_state_dict", require_lora_names=True)
        if len(roundtrip_state) != len(trainable_state):
            raise CheckpointExportError("Wrapper round-trip tensor-count verification failed")
        if any(tensor.device.type != "cpu" for tensor in roundtrip_state.values()):
            raise CheckpointExportError("Wrapper round-trip contains a non-CPU tensor")
        roundtrip_sha = tensor_state_sha256(roundtrip_state)
        if roundtrip_sha != lora_sha:
            raise CheckpointExportError(
                f"Wrapper round-trip strong hash mismatch: expected={lora_sha} actual={roundtrip_sha}"
            )

        output_file_sha = file_sha256(output_temporary)
        summary: dict[str, Any] = {
            "schema": SUMMARY_SCHEMA,
            "protocol": WRAPPER_PROTOCOL,
            "source_path": str(source),
            "source_file_sha256": source_file_sha,
            "source_schema": source_schema,
            "source_protocol": source_protocol,
            "source_train_mode": source_train_mode,
            "source_step": source_step,
            "source_lora_state_sha256": lora_sha,
            "source_head_state_sha256": head_sha,
            "expected_source_file_sha256": expected_source_sha256,
            "expected_lora_state_sha256": expected_lora_sha256,
            "output_path": str(output),
            "output_file_sha256": output_file_sha,
            "output_schema": WRAPPER_SCHEMA,
            "trainable_state_sha256": roundtrip_sha,
            "lora_tensor_count": len(roundtrip_state),
            "expected_lora_tensors": expected_lora_tensors,
            "summary_path": str(summary_path),
            "roundtrip_verified": True,
        }
        summary_temporary = _write_json_temporary(summary_path, summary)

        _publish_temporary(output_temporary, output, overwrite=overwrite)
        output_published = True
        _publish_temporary(summary_temporary, summary_path, overwrite=overwrite)
        summary_temporary = None
    except Exception:
        if output_published and not overwrite:
            output.unlink(missing_ok=True)
        raise
    finally:
        output_temporary.unlink(missing_ok=True)
        if summary_temporary is not None:
            summary_temporary.unlink(missing_ok=True)

    return summary


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--summary-json",
        type=Path,
        default=None,
        help="JSON sidecar path (default: <output>.summary.json).",
    )
    parser.add_argument(
        "--expected-lora-tensors",
        type=int,
        default=DEFAULT_EXPECTED_LORA_TENSORS,
        help=f"Required LoRA tensor count (default: {DEFAULT_EXPECTED_LORA_TENSORS}).",
    )
    parser.add_argument(
        "--expected-source-sha256",
        default=None,
        help="Optional exact lowercase SHA-256 pin for the source checkpoint file.",
    )
    parser.add_argument(
        "--expected-lora-sha256",
        default=None,
        help="Optional exact lowercase SHA-256 pin for the source LoRA tensor state.",
    )
    parser.add_argument("--overwrite", action="store_true", help="Explicitly allow replacing both outputs.")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    summary = export_pose_free_lora_wrapper(
        args.source_checkpoint,
        args.output,
        summary_json=args.summary_json,
        expected_lora_tensors=args.expected_lora_tensors,
        expected_source_sha256=args.expected_source_sha256,
        expected_lora_sha256=args.expected_lora_sha256,
        overwrite=args.overwrite,
    )
    print(json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

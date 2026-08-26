#!/usr/bin/env python3
"""RPC exporter for paired native/control candidate-support auditing.

The released InternNav System2 prompt and System1 weights stay frozen.  The
server runs both frozen System1 arms from the same explicit initial diffusion
noise and returns compact deployable features as a binary NPZ attachment.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import io
import json
import logging
import os
import re
import signal
import sys
import types
from collections import OrderedDict
from concurrent import futures
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

os.environ.setdefault("USE_TF", "0")
os.environ.setdefault("TRANSFORMERS_NO_TF", "1")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

FJL_ROOT = Path(os.environ.get("HEATMAPVLN_FJL_ROOT", "/mnt/afs/liwenhao/agent/370910109"))
HEATMAP_REPO = Path(os.environ.get("HEATMAPVLN_REPO", str(FJL_ROOT / "HeatmapVLN")))
INTERNNAV_REPO = Path(os.environ.get("INTERNNAV_REPO", str(FJL_ROOT / "InternNav")))

# HeatmapVLN owns the shared RPC protocol; InternNav owns the native model.
sys.path.insert(0, str(INTERNNAV_REPO))
sys.path.insert(0, str(HEATMAP_REPO))

import grpc
import numpy as np
import torch
from PIL import Image
from locked_rpc_protocol import (
    HEATMAPVLN_RPC_PROTOCOL_VERSION,
    HEATMAPVLN_RPC_SAMPLING_FIELD,
    validate_rpc_sampling_metadata,
)
from scripts.training.frozen_heatmap_checkpoint import (
    load_frozen_heatmap_checkpoint,
)
from scripts.training.utils import _normalize_state_key, load_config
from scripts.evaluation.candidate_support_audit import (
    AUDIT_SCHEMA_VERSION,
    compact_array_manifest,
    sha256_bytes,
    validate_compact_arrays,
)
from src.data.trajectory_utils import compute_history_rel_poses
from vla_rpc.core.image import decode_jpeg_to_rgb
from vla_rpc.proto import vla_pb2, vla_pb2_grpc


LOGGER = logging.getLogger("candidate-support-rpc-server")
PROTO_VERSION = HEATMAPVLN_RPC_PROTOCOL_VERSION
CONTROL_PROTO_VERSION = "heatmap-control-eval-v1"
GT_POSE_PROVIDER = "habitat_gt_c2w"
AMB3R_POSE_PROVIDER = "amb3r_vo_da3"
CANDIDATE_EXPORT_PROTO_VERSION = "paired-candidate-export-v1"
CANDIDATE_BLOB_NAME = "candidate_audit/compact_features.npz"
DEFAULT_IMAGE_TOKEN = "<image>"
MAX_STEPS = 8
MAX_LOCAL_STEPS = 4
VIEW_ORDER = ("front", "right", "back", "left")
CANDIDATE_SOURCE_RELATIVE_PATHS = (
    "scripts/evaluation/candidate_support_audit.py",
    "scripts/training/model_builder.py",
    "src/models/action/heatmap_control.py",
    "src/models/action/nextdit/__init__.py",
    "src/models/action/nextdit/components.py",
    "src/models/action/nextdit/nextdit_crossattn.py",
    "src/models/action/nextdit/nextdit_traj.py",
    "src/models/action/nextdit_action_head.py",
    "src/models/heatmap/structured_heatmap_tokenizer.py",
    "src/models/pipeline.py",
)

NATIVE_PROMPT = (
    "You are an autonomous navigation assistant. Your task is to <instruction>. "
    "Where should you go next to stay on track? Please output the next waypoint's "
    "coordinates in the image. Please output STOP when you have successfully completed the task."
)
NATIVE_CONJUNCTION = "you can see "
NATIVE_ACTIONS = OrderedDict(
    {
        "STOP": [0],
        "↑": [1],
        "←": [2],
        "→": [3],
        "↓": [5],
    }
)


def load_evaluation_model_config(
    config_path: str | Path,
    model_path: str | Path,
) -> dict[str, Any]:
    """Load the locked inference config without imposing TrainConfig fields.

    The evaluation plan intentionally has no dataset root, optimizer, or log
    directory.  Those are training concerns.  Keep the inference boundary
    fail-closed by validating the model/control contract needed by this RPC
    runtime explicitly instead of padding the YAML with unused train fields.
    """
    cfg = load_config(str(config_path), validate=False)
    native_model_path = str(Path(model_path).expanduser().resolve())
    model_cfg = cfg.setdefault("model", {})
    llm_cfg = model_cfg.setdefault("llm", {})
    llm_cfg["model_path"] = native_model_path
    nextdit_cfg = model_cfg.setdefault("action_head", {}).setdefault(
        "nextdit", {}
    )
    nextdit_cfg["internnav_model_path"] = native_model_path
    nextdit_cfg["internnav_system1_path"] = ""
    control_cfg = nextdit_cfg.setdefault("heatmap_control", {})
    heatmap_cfg = model_cfg.get("heatmap", {})
    trajectory_cfg = cfg.get("data", {}).get("trajectory", {})

    checks = {
        "pipeline": model_cfg.get("type") == "vln_pipeline",
        "native_llm": llm_cfg.get("model_path") == native_model_path,
        "native_system1": (
            nextdit_cfg.get("internnav_model_path") == native_model_path
            and nextdit_cfg.get("internnav_system1_path") == ""
        ),
        "action_head": bool(model_cfg.get("action_head", {}).get("enable")),
        "nextdit": bool(nextdit_cfg.get("enabled")),
        "twelve_layers": int(nextdit_cfg.get("dit_layers", -1)) == 12,
        "single_view_heatmap": (
            bool(heatmap_cfg.get("enable"))
            and heatmap_cfg.get("input_mode") == "internnav_single_view"
        ),
        "minus_z_pose": heatmap_cfg.get("history_pose_convention")
        == "habitat_c2w_minus_z__forward_left_cos_yaw_sin_yaw__v1",
        "native_prompt": trajectory_cfg.get("system2_sft_protocol")
        == "internnav",
        "no_lora": not bool(llm_cfg.get("use_lora")),
        "control_enabled": bool(control_cfg.get("enabled")),
    }
    failed = sorted(name for name, valid in checks.items() if not valid)
    if failed:
        raise RuntimeError(
            "Evaluation model config contract failed: "
            f"failed={failed} checks={checks}"
        )

    expected_control = {
        "schema_version": "heatmap-control-v1",
        "token_dim": 128,
        "control_dim": 128,
        "num_heads": 4,
        "coarse_size": 8,
        "temporal_layers": 1,
        "temporal_heads": 4,
        "temporal_ffn_dim": 512,
        "dropout": 0.0,
        "age_normalizer_steps": 32.0,
    }
    mismatches = {
        name: {"expected": expected, "actual": control_cfg.get(name)}
        for name, expected in expected_control.items()
        if control_cfg.get(name) != expected
    }
    if mismatches:
        raise RuntimeError(
            "Evaluation heatmap-control architecture mismatch: "
            f"{mismatches}"
        )
    return cfg


class ActionCode:
    STOP = 0
    FORWARD = 1
    LEFT = 2
    RIGHT = 3
    LOOKUP = 4
    LOOKDOWN = 5


def _install_numpy_legacy_aliases() -> None:
    if not hasattr(np, "float"):
        np.float = np.float64
    if not hasattr(np, "int"):
        np.int = np.int64
    if not hasattr(np, "bool"):
        np.bool = np.bool_


def _split_and_clean(text: str) -> list[str]:
    parts = re.split(r"(<image>)", text)
    result: list[str] = []
    for part in parts:
        if part == DEFAULT_IMAGE_TOKEN:
            result.append(part)
        else:
            cleaned = part.replace("\n", "").strip()
            if cleaned:
                result.append(cleaned)
    return result


def _content_from_prompt(prompt: str, images: list[Image.Image], start_index: int = 0) -> tuple[list[dict], int]:
    content: list[dict] = []
    image_index = start_index
    for part in _split_and_clean(prompt):
        if part == DEFAULT_IMAGE_TOKEN:
            if not 0 <= image_index < len(images):
                raise RuntimeError(
                    f"Native prompt/image mismatch: requested index {image_index}, images={len(images)}"
                )
            content.append({"type": "image", "image": images[image_index]})
            image_index += 1
        else:
            content.append({"type": "text", "text": part})
    return content, image_index


def build_native_messages(
    instruction: str,
    history_front: list[Image.Image],
    current_front: Image.Image,
) -> tuple[list[dict], list[Image.Image]]:
    """Reproduce InternNav's front-only R2R System2 prompt deterministically."""
    prompt = NATIVE_PROMPT.replace("<instruction>.", instruction)
    if history_front:
        placeholders = (DEFAULT_IMAGE_TOKEN + "\n") * len(history_front)
        prompt += f" These are your historical observations: {placeholders}."
    prompt += f" {NATIVE_CONJUNCTION}{DEFAULT_IMAGE_TOKEN}."
    images = list(history_front) + [current_front]
    content, consumed = _content_from_prompt(prompt, images)
    if consumed != len(images):
        raise RuntimeError(f"Native prompt consumed {consumed}/{len(images)} images")
    return [{"role": "user", "content": content}], images


def append_native_lookdown_turn(
    messages: list[dict],
    images: list[Image.Image],
    first_output: str,
    lookdown: Image.Image,
) -> tuple[list[dict], list[Image.Image]]:
    """Reproduce the official second conversational turn after a down-arrow."""
    messages = copy.deepcopy(messages)
    messages.append({"role": "assistant", "content": [{"type": "text", "text": first_output}]})
    images = list(images) + [lookdown]
    content, consumed = _content_from_prompt(
        f"{NATIVE_CONJUNCTION}{DEFAULT_IMAGE_TOKEN}.",
        images,
        start_index=len(images) - 1,
    )
    if consumed != len(images):
        raise RuntimeError(f"Native lookdown turn consumed {consumed}/{len(images)} images")
    messages.append({"role": "user", "content": content})
    return messages, images


def parse_native_actions(output: str) -> list[int]:
    pattern = "|".join(re.escape(token) for token in NATIVE_ACTIONS)
    matches = re.findall(pattern, output or "")
    return [action for token in matches for action in NATIVE_ACTIONS[token]]


def _finalize_local_actions(actions: list[int]) -> list[int]:
    actions = list(actions)
    if len(actions) < MAX_STEPS:
        actions += [ActionCode.STOP] * (MAX_STEPS - len(actions))
    return [int(action) for action in actions[:MAX_LOCAL_STEPS]]


def _pil_from_blob(
    blob: vla_pb2.BinaryBlob,
    image_size: tuple[int, int] | None = None,
) -> Image.Image:
    array = decode_jpeg_to_rgb(blob.data)
    image = Image.fromarray(array.astype(np.uint8)).convert("RGB")
    if image_size is not None and image.size != image_size:
        image = image.resize(image_size)
    return image


def _blobs_by_name(blobs) -> dict[str, vla_pb2.BinaryBlob]:
    result = {blob.name: blob for blob in blobs}
    if len(result) != len(blobs):
        raise RuntimeError("Duplicate RPC blob name")
    return result


def _trajectory_summary(trajectory: torch.Tensor) -> str:
    if trajectory.ndim != 3 or trajectory.shape[-1] < 2:
        return f"trajectory_shape={tuple(trajectory.shape)}"
    deltas = trajectory.float().detach().cpu().numpy().copy()
    deltas[:, :, :2] /= 4.0
    paths = np.cumsum(deltas[:, :, :2], axis=1)
    paths = np.concatenate(
        [np.zeros((paths.shape[0], 1, 2), dtype=paths.dtype), paths], axis=1
    )
    mean_path = paths.mean(axis=0)
    endpoint = mean_path[-1]
    direct = float(np.linalg.norm(endpoint))
    path_len = float(np.linalg.norm(np.diff(mean_path, axis=0), axis=1).sum())
    return (
        f"native_traj_goal=({endpoint[0]:.2f},{endpoint[1]:.2f}), "
        f"direct={direct:.2f}, path_len={path_len:.2f}"
    )


@contextmanager
def _deterministic_torch_seed(device: torch.device, seed: int) -> Iterator[None]:
    devices: list[int] = []
    if device.type == "cuda":
        devices = [int(device.index if device.index is not None else torch.cuda.current_device())]
    with torch.random.fork_rng(devices=devices, enabled=True):
        torch.manual_seed(int(seed))
        if device.type == "cuda":
            torch.cuda.manual_seed_all(int(seed))
        yield


def _patch_native_depth_bootstrap() -> None:
    """Avoid InternNav's obsolete external DAV2 bootstrap file.

    The complete 175-tensor RGB encoder is present in InternNav-Model and is
    loaded by ``from_pretrained``.  Upstream nevertheless tries to read a
    separate ``checkpoints/depth_anything_*.pth`` before loading safetensors.
    Constructing the identical module without that redundant bootstrap keeps
    the model native while allowing strict full-checkpoint loading.
    """
    from internnav.model.basemodel.internvla_n1 import internvla_n1_arch as native_arch

    # Importing ``internnav.model.encoder`` executes an eager package __init__
    # that imports the optional, absent LongCLIP submodule before DAV2 can be
    # reached.  The encoder directory is otherwise a normal package root and
    # DAV2 has no dependency on those eager exports.  Register only that
    # package namespace so Python can load the untouched DAV2 implementation.
    encoder_package = "internnav.model.encoder"
    if encoder_package not in sys.modules:
        encoder_module = types.ModuleType(encoder_package)
        encoder_module.__path__ = [str(INTERNNAV_REPO / "internnav/model/encoder")]
        encoder_module.__package__ = encoder_package
        sys.modules[encoder_package] = encoder_module
        LOGGER.info("Bypassed optional LongCLIP encoder exports for native DAV2 import")

    from internnav.model.encoder.depth_anything.depth_anything_v2.dpt import DepthAnythingV2
    from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
    from internnav.model.basemodel.internvla_n1.nextdit_crossattn_traj import (
        NextDiTCrossAttn,
        NextDiTCrossAttnConfig,
    )

    def build_eval_traj_dit(_config):
        # Upstream enables training-only gradient checkpointing while building
        # the module.  Its legacy hook is incompatible with Diffusers 0.36 in
        # qwen25.  It has no effect under eval/no_grad, so construct the exact
        # same architecture with that runtime-only flag disabled.
        dit = NextDiTCrossAttn(
            NextDiTCrossAttnConfig(
                latent_embedding_size=native_arch.LatentEmbSize,
                # InternNav was trained against the earlier Diffusers Lumina
                # SwiGLU rule, whose 4*dim input is reduced by 2/3 before
                # rounding.  Diffusers 0.36 removed that implicit reduction;
                # make it explicit to recover the checkpoint's 1024-wide FFN.
                ffn_dim_multiplier=2.0 / 3.0,
                _gradient_checkpointing=False,
            )
        )
        return dit, FlowMatchEulerDiscreteScheduler()

    def build_depthanythingv2_from_full_checkpoint(_config):
        model = DepthAnythingV2(
            encoder="vits",
            features=64,
            out_channels=[48, 96, 192, 384],
        )
        return model.pretrained

    native_arch.build_depthanythingv2 = build_depthanythingv2_from_full_checkpoint
    native_arch.build_traj_dit = build_eval_traj_dit
    LOGGER.info(
        "Installed native construction shims: DAV2 weights must come from full "
        "safetensors; eval-only NextDiT gradient checkpointing disabled"
    )


def _checkpoint_index(model_path: Path) -> tuple[dict[str, str], list[str]]:
    index_path = model_path / "model.safetensors.index.json"
    document = json.loads(index_path.read_text(encoding="utf-8"))
    weight_map = document.get("weight_map")
    if not isinstance(weight_map, dict) or not weight_map:
        raise RuntimeError(f"Invalid native checkpoint index: {index_path}")
    shards = sorted(set(weight_map.values()))
    for shard in shards:
        path = model_path / shard
        if not path.is_file() or path.stat().st_size <= 0:
            raise FileNotFoundError(path)
    return weight_map, shards


def _file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_candidate_source_manifest(repo_root: str | Path) -> dict[str, str]:
    """Hash the exact candidate-generation source closure, failing early."""

    root = Path(repo_root).expanduser().resolve(strict=True)
    if not root.is_dir():
        raise NotADirectoryError(root)
    result: dict[str, str] = {}
    for relative in CANDIDATE_SOURCE_RELATIVE_PATHS:
        path = (root / relative).resolve()
        try:
            path.relative_to(root)
        except ValueError as exc:
            raise RuntimeError(
                f"candidate provenance path escapes repository: {relative}"
            ) from exc
        if not path.is_file():
            raise FileNotFoundError(
                f"candidate provenance source is missing: {path}"
            )
        result[relative] = _file_sha256(path)
    return result


def _json_safe(value: Any) -> Any:
    """Convert scheduler/config values into deterministic JSON primitives."""
    if value is None or isinstance(value, (str, bool, int, float)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in sorted(value.items())}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_safe(value.item())
        except (TypeError, ValueError):
            pass
    return repr(value)


def _tensor_float32(tensor: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(tensor.detach().float().cpu().numpy())


def _tensor_float16(tensor: torch.Tensor) -> np.ndarray:
    return np.ascontiguousarray(tensor.detach().float().cpu().numpy().astype(np.float16))


def _tensor_bfloat16_bits(tensor: torch.Tensor) -> np.ndarray:
    if tensor.dtype != torch.bfloat16:
        raise TypeError(
            f"Expected native bf16 inference tensor, got {tensor.dtype}"
        )
    value = tensor.detach().contiguous().view(torch.uint16)
    return np.ascontiguousarray(value.cpu().numpy())


def _pack_compact_arrays(
    arrays: dict[str, Any],
) -> tuple[bytes, dict[str, Any]]:
    validated = validate_compact_arrays(arrays)
    buffer = io.BytesIO()
    np.savez_compressed(buffer, **validated)
    payload = buffer.getvalue()
    manifest = compact_array_manifest(validated)
    manifest.update(
        {
            "blob_name": CANDIDATE_BLOB_NAME,
            "blob_mime_type": "application/x-npz",
            "blob_bytes": len(payload),
            "blob_sha256": sha256_bytes(payload),
        }
    )
    return payload, manifest


def _load_weights_only(path: Path) -> dict[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError("Control evaluation requires weights_only checkpoint loading") from exc
    if not isinstance(payload, dict):
        raise TypeError(f"Control checkpoint must be a mapping: {path}")
    return payload


def _control_parameter_map(model: torch.nn.Module) -> dict[str, torch.nn.Parameter]:
    expected = {
        name: parameter
        for name, parameter in model.named_parameters()
        if name.startswith("heatmap_tokenizer.") or ".heatmap_control." in name
    }
    adapters = tuple(model.nextdit_action_head.heatmap_control_adapters())
    if len(adapters) != 12:
        raise RuntimeError(f"Expected 12 heatmap-control adapters, got {len(adapters)}")
    if not expected:
        raise RuntimeError("Heatmap tokenizer/control parameters were not constructed")
    return expected


def _normalized_floating_state(
    raw_state: Any,
    *,
    label: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(raw_state, dict) or not raw_state:
        raise RuntimeError(f"Control checkpoint {label} is empty")
    state: dict[str, torch.Tensor] = {}
    raw_by_name: dict[str, str] = {}
    for raw_name, value in raw_state.items():
        name = _normalize_state_key(str(raw_name))
        if name in state:
            raise RuntimeError(
                f"Duplicate normalized {label} keys: {raw_by_name[name]!r}, {raw_name!r}"
            )
        if not torch.is_tensor(value):
            raise TypeError(f"{label} value is not a tensor: {raw_name}")
        if value.layout != torch.strided or not value.is_floating_point():
            raise TypeError(f"{label} tensor must be dense strided floating point: {raw_name}")
        if not bool(torch.isfinite(value.float()).all()):
            raise RuntimeError(f"Non-finite {label} tensor: {raw_name}")
        state[name] = value
        raw_by_name[name] = str(raw_name)
    return state


def _require_exact_ema_deployment_state(
    trainable_state: dict[str, torch.Tensor],
    ema_state_dict: Any,
) -> None:
    if not isinstance(ema_state_dict, dict):
        raise RuntimeError("EMA deployment checkpoint is missing ema_state_dict")
    shadow = _normalized_floating_state(
        ema_state_dict.get("shadow"),
        label="ema_state_dict.shadow",
    )
    if set(trainable_state) != set(shadow):
        raise RuntimeError(
            "EMA deployment state key mismatch: "
            f"trainable_only={sorted(set(trainable_state) - set(shadow))[:8]} "
            f"shadow_only={sorted(set(shadow) - set(trainable_state))[:8]}"
        )
    mismatches = []
    for name, value in trainable_state.items():
        reference = shadow[name]
        if (
            value.shape != reference.shape
            or value.dtype != reference.dtype
            or not torch.equal(value, reference)
        ):
            mismatches.append(name)
    if mismatches:
        raise RuntimeError(
            "trainable_state_dict is not the exact EMA shadow deployment state: "
            f"{mismatches[:8]}"
        )


def load_control_ema_deploy_checkpoint(
    model: torch.nn.Module,
    checkpoint_path: str | Path,
    expected_sha256: str,
    frozen_heatmap_dependency: dict[str, Any],
    expected_native_model_path: str | Path,
    expected_native_manifest_path: str | Path,
    expected_native_manifest_sha256: str,
) -> dict[str, Any]:
    """Strictly load the checkpoint's EMA deployment entry and nothing else."""
    path = Path(checkpoint_path).expanduser().resolve(strict=True)
    if not re.fullmatch(r"[0-9a-f]{64}", str(expected_sha256)):
        raise ValueError("control checkpoint SHA256 must be 64 lowercase hex characters")
    actual_sha256 = _file_sha256(path)
    if actual_sha256 != expected_sha256:
        raise RuntimeError(
            "Control checkpoint SHA256 mismatch: "
            f"expected={expected_sha256} actual={actual_sha256}"
        )
    payload = _load_weights_only(path)
    if type(payload.get("epoch")) is not int or payload.get("epoch") != 3 or payload.get("batch") is not None:
        raise RuntimeError(
            "Control evaluation requires last complete epoch_003 "
            f"(epoch=3,batch=None), got epoch={payload.get('epoch')!r} "
            f"batch={payload.get('batch')!r}"
        )
    if type(payload.get("stage_idx")) is not int or payload.get("stage_idx") != 0:
        raise RuntimeError(
            "Control evaluation requires stage_idx=0, "
            f"got {payload.get('stage_idx')!r}"
        )
    saved_config = payload.get("config")
    if not isinstance(saved_config, dict):
        raise RuntimeError("Control checkpoint is missing its training config")
    saved_model = saved_config.get("model")
    if not isinstance(saved_model, dict):
        raise RuntimeError("Control checkpoint config.model is missing")
    saved_nextdit = saved_model.get("action_head", {}).get("nextdit", {})
    saved_control = saved_nextdit.get("heatmap_control", {})
    expected_control_architecture = {
        "enabled": True,
        "schema_version": "heatmap-control-v1",
        "token_dim": 128,
        "control_dim": 128,
        "num_heads": 4,
        "coarse_size": 8,
        "temporal_layers": 1,
        "temporal_heads": 4,
        "temporal_ffn_dim": 512,
        "dropout": 0.0,
        "age_normalizer_steps": 32.0,
    }
    mismatches = {
        name: {"expected": value, "actual": saved_control.get(name)}
        for name, value in expected_control_architecture.items()
        if saved_control.get(name) != value
    }
    if int(saved_nextdit.get("dit_layers", -1)) != 12:
        mismatches["dit_layers"] = {
            "expected": 12,
            "actual": saved_nextdit.get("dit_layers"),
        }
    if mismatches:
        raise RuntimeError(f"Control checkpoint architecture mismatch: {mismatches}")
    dependency_sha = frozen_heatmap_dependency["checkpoint_sha256"]
    if saved_control.get("heatmap_checkpoint_sha256") != dependency_sha:
        raise RuntimeError(
            "Control checkpoint was trained against a different frozen heatmap: "
            f"saved={saved_control.get('heatmap_checkpoint_sha256')!r} "
            f"current={dependency_sha}"
        )
    saved_dependency = saved_config.get("runtime", {}).get("frozen_heatmap_dependency")
    if not isinstance(saved_dependency, dict):
        raise RuntimeError("Control checkpoint lacks runtime.frozen_heatmap_dependency")
    dependency_fields = (
        "schema_version",
        "dependency_type",
        "checkpoint_sha256",
        "state_key",
        "target_module",
        "tensor_count",
        "parameter_names",
        "parameter_shapes",
        "frozen",
    )
    dependency_mismatches = {
        name: {
            "saved": saved_dependency.get(name),
            "current": frozen_heatmap_dependency.get(name),
        }
        for name in dependency_fields
        if saved_dependency.get(name) != frozen_heatmap_dependency.get(name)
    }
    if dependency_mismatches:
        raise RuntimeError(
            f"Frozen heatmap dependency contract mismatch: {dependency_mismatches}"
        )
    expected_native = Path(expected_native_model_path).expanduser().resolve()
    for label, raw_path in (
        ("System2", saved_model.get("llm", {}).get("model_path")),
        ("System1", saved_nextdit.get("internnav_model_path")),
    ):
        if not raw_path or Path(str(raw_path)).expanduser().resolve() != expected_native:
            raise RuntimeError(
                f"Control checkpoint {label} path mismatch: {raw_path!r} != {expected_native}"
            )
    expected_native_manifest = Path(expected_native_manifest_path).expanduser().resolve()
    expected_native_dependency = {
        "schema": "native-internnav-checkpoint-v1",
        "model_path": str(expected_native),
        "manifest_path": str(expected_native_manifest),
        "manifest_sha256": str(expected_native_manifest_sha256),
        "file_count": 14,
        "verified": True,
    }
    saved_native_dependency = saved_config.get("runtime", {}).get(
        "native_internnav_dependency"
    )
    if not isinstance(saved_native_dependency, dict):
        raise RuntimeError("Control checkpoint lacks runtime.native_internnav_dependency")
    if set(saved_native_dependency) != set(expected_native_dependency):
        raise RuntimeError(
            "Control checkpoint native InternNav dependency fields are not exact: "
            f"missing={sorted(set(expected_native_dependency) - set(saved_native_dependency))} "
            f"extra={sorted(set(saved_native_dependency) - set(expected_native_dependency))}"
        )
    native_dependency_mismatches = {
        name: {
            "expected": expected,
            "actual": saved_native_dependency.get(name),
        }
        for name, expected in expected_native_dependency.items()
        if saved_native_dependency.get(name) != expected
        or type(saved_native_dependency.get(name)) is not type(expected)
    }
    if native_dependency_mismatches:
        raise RuntimeError(
            "Control checkpoint native InternNav dependency mismatch: "
            f"{native_dependency_mismatches}"
        )
    semantics = payload.get("weight_semantics")
    if not isinstance(semantics, dict) or semantics.get("trainable_state_dict") != "ema":
        raise RuntimeError(
            "Control evaluation requires trainable_state_dict with EMA deployment semantics"
        )
    if str(payload.get("stage_name")) != "heatmap_system1_control":
        raise RuntimeError(
            f"Unexpected control checkpoint stage: {payload.get('stage_name')!r}"
        )
    state = _normalized_floating_state(
        payload.get("trainable_state_dict"),
        label="trainable_state_dict",
    )
    _require_exact_ema_deployment_state(state, payload.get("ema_state_dict"))

    targets = _control_parameter_map(model)
    missing = sorted(set(targets) - set(state))
    unexpected = sorted(set(state) - set(targets))
    if missing or unexpected:
        raise RuntimeError(
            "Control EMA parameter coverage is not exact: "
            f"missing={missing[:8]} unexpected={unexpected[:8]}"
        )
    shape_mismatches = {
        name: (tuple(state[name].shape), tuple(targets[name].shape))
        for name in targets
        if tuple(state[name].shape) != tuple(targets[name].shape)
    }
    if shape_mismatches:
        raise RuntimeError(
            f"Control EMA parameter shape mismatch: {dict(list(shape_mismatches.items())[:8])}"
        )
    converted: dict[str, torch.Tensor] = {}
    for name, target in targets.items():
        value = state[name].detach().to(device=target.device, dtype=target.dtype).contiguous()
        if not bool(torch.isfinite(value.float()).all()):
            raise RuntimeError(f"Non-finite control EMA tensor: {name}")
        converted[name] = value
    with torch.no_grad():
        for name, target in targets.items():
            target.copy_(converted[name])
        for name, target in targets.items():
            if not torch.equal(target.detach(), converted[name]):
                raise RuntimeError(f"Control EMA post-copy verification failed: {name}")
    return {
        "checkpoint_path": str(path),
        "checkpoint_sha256": actual_sha256,
        "state_key": "trainable_state_dict",
        "weight_semantics": "ema",
        "tensor_count": len(targets),
        "parameter_count": sum(parameter.numel() for parameter in targets.values()),
        "adapter_count": 12,
        "native_internnav_dependency": expected_native_dependency,
    }


def _validate_capture_metadata(
    payload: dict[str, Any],
    num_history: int,
) -> dict[str, Any]:
    """Validate provider-independent causal capture-step metadata."""
    history_steps = np.asarray(payload.get("history_capture_steps"), dtype=np.int64)
    ages = np.asarray(payload.get("history_age_steps"), dtype=np.int64)
    current_step = int(payload.get("current_capture_step", -1))
    if history_steps.shape != (num_history,) or ages.shape != (num_history,):
        raise ValueError("history capture steps/ages do not match num_history")
    if current_step < 0 or np.any(history_steps < 0) or np.any(history_steps > current_step):
        raise ValueError("capture steps must be non-negative and causal")
    expected_ages = current_step - history_steps
    if not np.array_equal(ages, expected_ages) or np.any(ages < 0):
        raise ValueError(
            f"history ages are inconsistent: expected={expected_ages.tolist()} got={ages.tolist()}"
        )
    return {
        "current_capture_step": current_step,
        "history_capture_steps": history_steps,
        "history_age_steps": ages,
    }


def _validate_gt_pose_metadata(
    payload: dict[str, Any],
    num_history: int,
) -> dict[str, Any]:
    """Validate the legacy Habitat-c2w provider without changing its geometry."""
    forbidden = sorted(
        key
        for key in (
            "history_rel_poses",
            "pose_ready",
            "vo_current_frame_id",
            "vo_history_frame_ids",
            "vo_provider_phase",
            "vo_trajectory_revision",
        )
        if key in payload
    )
    if forbidden:
        raise ValueError(
            f"{GT_POSE_PROVIDER} payload may not contain VO metadata: {forbidden}"
        )
    if payload.get("control_proto_v") != CONTROL_PROTO_VERSION:
        raise ValueError(
            f"control protocol mismatch: {payload.get('control_proto_v')!r}"
        )
    current_pose = np.asarray(payload.get("current_c2w"), dtype=np.float32)
    raw_history_poses = payload.get("history_c2w")
    if num_history == 0 and raw_history_poses == []:
        history_poses = np.empty((0, 4, 4), dtype=np.float32)
    else:
        history_poses = np.asarray(raw_history_poses, dtype=np.float32)
    if current_pose.shape != (4, 4) or not np.isfinite(current_pose).all():
        raise ValueError(f"current_c2w must be finite [4,4], got {current_pose.shape}")
    if history_poses.shape != (num_history, 4, 4) or not np.isfinite(history_poses).all():
        raise ValueError(
            f"history_c2w must be finite [{num_history},4,4], got {history_poses.shape}"
        )
    all_poses = np.concatenate((current_pose[None], history_poses), axis=0)
    expected_last_row = np.asarray((0.0, 0.0, 0.0, 1.0), dtype=np.float32)
    for pose_index, pose in enumerate(all_poses):
        if not np.allclose(pose[3], expected_last_row, atol=1e-4, rtol=0.0):
            raise ValueError(f"c2w[{pose_index}] has an invalid homogeneous last row")
        rotation = pose[:3, :3]
        if not np.allclose(rotation.T @ rotation, np.eye(3), atol=2e-3, rtol=0.0):
            raise ValueError(f"c2w[{pose_index}] rotation is not orthonormal")
        if not np.isclose(np.linalg.det(rotation), 1.0, atol=2e-3, rtol=0.0):
            raise ValueError(f"c2w[{pose_index}] rotation determinant is not +1")
    history_rel = compute_history_rel_poses(
        list(history_poses),
        current_pose,
        camera_forward_axis="-z",
    )
    return {
        "current_c2w": current_pose,
        "history_c2w": history_poses,
        "history_rel_poses": np.ascontiguousarray(history_rel, dtype=np.float32),
        "pose_ready": True,
    }


def _strict_nonnegative_int(value: Any, label: str) -> int:
    if isinstance(value, (bool, np.bool_)) or not isinstance(value, (int, np.integer)):
        raise TypeError(f"{label} must be an integer")
    result = int(value)
    if result < 0:
        raise ValueError(f"{label} must be non-negative")
    return result


def _validate_amb3r_pose_metadata(
    payload: dict[str, Any],
    num_history: int,
) -> dict[str, Any]:
    """Validate non-privileged relative poses supplied by online AMB3R-VO."""
    forbidden = sorted(key for key in ("current_c2w", "history_c2w") if key in payload)
    if forbidden:
        raise ValueError(
            f"{AMB3R_POSE_PROVIDER} payload must not contain privileged c2w fields: {forbidden}"
        )
    if type(payload.get("pose_ready")) is not bool:
        raise TypeError("amb3r_vo_da3 pose_ready must be a JSON boolean")
    pose_ready = bool(payload["pose_ready"])

    current_frame_id = _strict_nonnegative_int(
        payload.get("vo_current_frame_id"),
        "vo_current_frame_id",
    )
    raw_history_frame_ids = payload.get("vo_history_frame_ids")
    if not isinstance(raw_history_frame_ids, list):
        raise TypeError("vo_history_frame_ids must be a JSON list")
    history_frame_ids = np.asarray(
        [
            _strict_nonnegative_int(value, f"vo_history_frame_ids[{index}]")
            for index, value in enumerate(raw_history_frame_ids)
        ],
        dtype=np.int64,
    )
    if history_frame_ids.shape != (num_history,):
        raise ValueError(
            "vo_history_frame_ids does not match num_history: "
            f"expected={num_history} got={history_frame_ids.shape}"
        )
    if np.any(history_frame_ids > current_frame_id):
        raise ValueError("AMB3R history frame IDs must be no later than current")
    if history_frame_ids.size > 1 and np.any(np.diff(history_frame_ids) < 0):
        raise ValueError(
            "AMB3R history frame IDs must be chronological (non-decreasing)"
        )

    provider_phase = payload.get("vo_provider_phase")
    if not isinstance(provider_phase, str) or not provider_phase.strip():
        raise ValueError("vo_provider_phase must be a non-empty string")
    trajectory_revision = _strict_nonnegative_int(
        payload.get("vo_trajectory_revision"),
        "vo_trajectory_revision",
    )

    if not pose_ready:
        if "history_rel_poses" in payload:
            raise ValueError(
                "AMB3R warmup payload must omit history_rel_poses while pose_ready=false"
            )
        history_rel = np.empty((0, 4), dtype=np.float32)
    else:
        if "history_rel_poses" not in payload:
            raise ValueError("AMB3R ready payload requires history_rel_poses")
        raw_history_rel = payload["history_rel_poses"]
        if not isinstance(raw_history_rel, list) or any(
            not isinstance(row, list)
            or any(
                isinstance(value, bool)
                or not isinstance(value, (int, float))
                for value in row
            )
            for row in raw_history_rel
        ):
            raise TypeError(
                "AMB3R history_rel_poses must be a JSON list of numeric rows"
            )
        if num_history == 0 and raw_history_rel == []:
            history_rel = np.empty((0, 4), dtype=np.float32)
        else:
            history_rel = np.asarray(raw_history_rel, dtype=np.float32)
        if history_rel.shape != (num_history, 4) or not np.isfinite(history_rel).all():
            raise ValueError(
                "AMB3R history_rel_poses must be finite "
                f"[{num_history},4], got {history_rel.shape}"
            )
        yaw_norm = np.linalg.norm(history_rel[:, 2:4], axis=1)
        if not np.allclose(yaw_norm, 1.0, atol=2e-3, rtol=0.0):
            raise ValueError(
                "AMB3R history_rel_poses yaw must be encoded as unit (cos,sin)"
            )
        history_rel = np.ascontiguousarray(history_rel, dtype=np.float32)

    return {
        "history_rel_poses": history_rel,
        "pose_ready": pose_ready,
        "vo_current_frame_id": current_frame_id,
        "vo_history_frame_ids": history_frame_ids,
        "vo_provider_phase": provider_phase.strip(),
        "vo_trajectory_revision": trajectory_revision,
    }


def validate_history_metadata(payload: dict[str, Any], num_history: int) -> dict[str, Any]:
    """Validate one unambiguous GT-c2w or external AMB3R pose payload.

    Requests predating the provider field remain GT-compatible.  AMB3R must
    always opt in explicitly, is forbidden from carrying simulator c2w, and
    can mark its warmup state unavailable without blocking native System1.
    """
    if payload.get("control_proto_v") != CONTROL_PROTO_VERSION:
        raise ValueError(
            f"control protocol mismatch: {payload.get('control_proto_v')!r}"
        )
    raw_provider = payload.get("pose_provider")
    provider_explicit = raw_provider is not None
    pose_provider = GT_POSE_PROVIDER if raw_provider is None else raw_provider
    if pose_provider == GT_POSE_PROVIDER:
        provider_metadata = _validate_gt_pose_metadata(payload, num_history)
    elif pose_provider == AMB3R_POSE_PROVIDER:
        provider_metadata = _validate_amb3r_pose_metadata(payload, num_history)
    else:
        raise ValueError(f"unsupported pose_provider: {pose_provider!r}")
    return {
        **_validate_capture_metadata(payload, num_history),
        **provider_metadata,
        "pose_provider": pose_provider,
        "pose_provider_explicit": provider_explicit,
    }


def _heatmap_control_input_ready(
    control_mode: str,
    num_history: int,
    metadata: dict[str, Any],
) -> bool:
    """Return whether pose-conditioned control may run for this request."""
    return (
        control_mode != "off"
        and int(num_history) > 0
        and metadata.get("pose_ready") is True
    )


def mismatch_heatmap_predictions(
    heatmap_logits: torch.Tensor,
    visibility_logits: torch.Tensor,
    history_mask: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Break spatial meaning before tokenization while preserving metadata.

    Cross-attention is invariant to a joint permutation of context tokens, so
    shuffling finalized tokens is not a valid causal control.  This ablation
    instead reverses valid history predictions and rolls their view direction
    by one, while keeping age/rank/yaw/mask metadata at the original slots.
    """
    mismatched_heatmaps = heatmap_logits.clone()
    mismatched_visibility = visibility_logits.clone()
    for batch_index in range(heatmap_logits.shape[0]):
        valid = torch.nonzero(history_mask[batch_index], as_tuple=False).flatten()
        if valid.numel() == 0:
            continue
        source = torch.flip(valid, dims=(0,))
        selected_heatmaps = heatmap_logits[batch_index, source]
        selected_visibility = visibility_logits[batch_index, source]
        mismatched_heatmaps[batch_index, valid] = torch.roll(
            selected_heatmaps,
            shifts=1,
            dims=1,
        )
        mismatched_visibility[batch_index, valid] = torch.roll(
            selected_visibility,
            shifts=1,
            dims=1,
        )
    return mismatched_heatmaps, mismatched_visibility


class HeatmapControlRuntime:
    def __init__(self, args: argparse.Namespace):
        _install_numpy_legacy_aliases()
        self.args = args
        self.model_path = Path(args.model_path).resolve()
        self.device = torch.device(f"cuda:{args.gpu_id}" if torch.cuda.is_available() else "cpu")
        self.require_deterministic_sampling = bool(args.require_deterministic_sampling)
        self.control_mode = str(args.control_mode)
        if self.control_mode != "on":
            raise ValueError(
                "candidate audit requires --control_mode on so both paired arms exist"
            )
        self.deployment_arm = str(args.deployment_arm)
        if self.deployment_arm not in {"native", "heatmap_control"}:
            raise ValueError(f"Unsupported deployment arm: {self.deployment_arm!r}")
        self.candidate_source_sha256 = build_candidate_source_manifest(HEATMAP_REPO)
        LOGGER.info(
            "Candidate source provenance verified: files=%d",
            len(self.candidate_source_sha256),
        )
        (
            self.model,
            self.processor,
            self.heatmap_dependency,
            self.control_dependency,
        ) = self._load_model()
        LOGGER.info("require_deterministic_sampling=%s", self.require_deterministic_sampling)
        LOGGER.info(
            "Candidate-support audit mode: native_front_only_system2=True "
            "native_frozen_system1=True control_mode=%s deployment_arm=%s adapters=12 "
            "vlm_image_size=384 lookdown_vlm_size=640x480 traj_image_size=224",
            self.control_mode,
            self.deployment_arm,
        )

    def _load_model(self):
        if not self.model_path.is_dir():
            raise FileNotFoundError(self.model_path)
        native_manifest = Path(self.args.native_model_manifest).expanduser().resolve(
            strict=True
        )
        expected_native_manifest_sha = str(self.args.native_model_manifest_sha256)
        if re.fullmatch(r"[0-9a-f]{64}", expected_native_manifest_sha) is None:
            raise ValueError("native model manifest SHA256 must be 64 lowercase hex")
        actual_native_manifest_sha = _file_sha256(native_manifest)
        if actual_native_manifest_sha != expected_native_manifest_sha:
            raise RuntimeError(
                "Native model manifest SHA256 mismatch: "
                f"expected={expected_native_manifest_sha} "
                f"actual={actual_native_manifest_sha}"
            )
        weight_map, shards = _checkpoint_index(self.model_path)
        lora_keys = [key for key in weight_map if "lora" in key.lower()]
        adapter_keys = [key for key in weight_map if "adapter" in key.lower()]
        if lora_keys or adapter_keys:
            raise RuntimeError(
                f"Native checkpoint unexpectedly contains LoRA/adapter keys: "
                f"lora={lora_keys[:3]} adapter={adapter_keys[:3]}"
            )
        if len(weight_map) != 1338 or len(shards) != 4:
            raise RuntimeError(
                f"Unexpected native checkpoint closure: tensors={len(weight_map)} shards={len(shards)}"
            )
        LOGGER.info(
            "Native InternNav checkpoint index verified: tensors=%d shards=%d lora=0 adapter=0",
            len(weight_map),
            len(shards),
        )

        cfg = load_evaluation_model_config(self.args.config, self.model_path)

        from scripts.training.model_builder import (
            assert_complete_internnav_system1_load,
            build_model,
        )

        model = build_model(cfg, device=str(self.device), verbose=True)
        model = model.to(self.device)
        model.qwen2_5_vl._load_model()
        model._ensure_heatmap_vln()
        required_system1 = assert_complete_internnav_system1_load(model, logger=LOGGER)
        LOGGER.info(
            "Native InternNav System1 strict load verified: tensors=%d",
            required_system1,
        )
        heatmap_dependency = load_frozen_heatmap_checkpoint(
            model,
            self.args.frozen_heatmap_checkpoint,
            self.args.frozen_heatmap_sha256,
        )
        LOGGER.info("Frozen heatmap strict load verified: %s", heatmap_dependency)
        control_dependency = load_control_ema_deploy_checkpoint(
            model,
            self.args.control_checkpoint,
            self.args.control_checkpoint_sha256,
            heatmap_dependency,
            self.model_path,
            self.args.native_model_manifest,
            self.args.native_model_manifest_sha256,
        )
        LOGGER.info("Control EMA strict load verified: %s", control_dependency)
        model.requires_grad_(False)
        model.eval()
        processor = model.qwen2_5_vl.processor
        processor.tokenizer.padding_side = "left"
        return model, processor, heatmap_dependency, control_dependency

    def _generate(self, messages: list[dict], images: list[Image.Image]):
        text = self.processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.processor(text=[text], images=images, return_tensors="pt").to(self.device)
        with torch.no_grad():
            output_ids = self.model.qwen2_5_vl.model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                use_cache=True,
                past_key_values=None,
                return_dict_in_generate=True,
            ).sequences
        output = self.processor.tokenizer.decode(
            output_ids[0][inputs.input_ids.shape[1] :],
            skip_special_tokens=True,
        )
        return output_ids, output, inputs

    def _build_heatmap_control(
        self,
        current_front: Image.Image,
        history_front: list[Image.Image],
        metadata: dict[str, Any],
        sampling_seed: int,
    ) -> dict[str, torch.Tensor] | None:
        if not _heatmap_control_input_ready(
            self.control_mode,
            len(history_front),
            metadata,
        ):
            return None
        fixed_history_slots = 8
        if len(history_front) > fixed_history_slots:
            raise ValueError("Heatmap control supports at most eight history slots")
        padding_count = fixed_history_slots - len(history_front)
        padding = [
            Image.new("RGB", current_front.size, color=(0, 0, 0))
            for _ in range(padding_count)
        ]
        flat_images = list(history_front) + padding + [current_front]
        encoded = self.processor.image_processor(
            images=flat_images,
            return_tensors="pt",
        )
        required = {"pixel_values", "image_grid_thw"}
        if required - set(encoded):
            raise RuntimeError(
                f"Heatmap image processor omitted {sorted(required - set(encoded))}"
            )
        if tuple(encoded["image_grid_thw"].shape) != (len(flat_images), 3):
            raise RuntimeError(
                "Heatmap image grouping mismatch: "
                f"expected={(len(flat_images), 3)} got={tuple(encoded['image_grid_thw'].shape)}"
            )
        valid_history_rel = np.asarray(
            metadata["history_rel_poses"],
            dtype=np.float32,
        )
        if valid_history_rel.shape != (len(history_front), 4):
            raise RuntimeError(
                "Validated history pose count diverged from history RGB count: "
                f"poses={valid_history_rel.shape} rgb={len(history_front)}"
            )
        history_rel = np.zeros((fixed_history_slots, 4), dtype=np.float32)
        history_rel[: len(history_front)] = valid_history_rel
        history_mask = torch.zeros((1, fixed_history_slots), dtype=torch.bool)
        history_mask[:, : len(history_front)] = True
        ages = torch.zeros((1, fixed_history_slots), dtype=torch.long)
        ages[:, : len(history_front)] = torch.from_numpy(
            metadata["history_age_steps"]
        )
        output = self.model._forward_frozen_single_view_heatmap(
            inputs={
                "pixel_values": encoded["pixel_values"],
                "image_grid_thw": encoded["image_grid_thw"],
            },
            num_histories=[fixed_history_slots],
            history_rel_poses=torch.from_numpy(history_rel).unsqueeze(0),
            explicit_history_mask=history_mask,
        )
        tokenizer = self.model.heatmap_tokenizer
        if tokenizer is None:
            raise RuntimeError("Heatmap tokenizer was not constructed")
        control_device = next(tokenizer.parameters()).device
        del sampling_seed
        heatmap_logits = output["heatmap_logits"].detach().to(
            device=control_device, dtype=torch.float32
        )
        visibility_logits = output["visibility"].detach().to(
            device=control_device, dtype=torch.float32
        )
        control_mask = history_mask.to(control_device)
        if self.control_mode == "mismatched":
            heatmap_logits, visibility_logits = mismatch_heatmap_predictions(
                heatmap_logits,
                visibility_logits,
                control_mask,
            )
        tokenized = tokenizer(
            heatmap_logits=heatmap_logits,
            visibility_logits=visibility_logits,
            history_mask=control_mask,
            history_age_steps=ages.to(device=control_device, dtype=torch.float32),
        )
        tokenized["sample_valid"] = tokenized["token_mask"].any(dim=1)
        # Preserve compact deployment-time inputs for the stage-0.5
        # identifiability probe. These are detached inference features, never
        # RGB frames or privileged simulator labels.
        tokenized["heatmap_logits"] = heatmap_logits
        tokenized["visibility_logits"] = visibility_logits
        tokenized["history_mask"] = control_mask
        tokenized["history_rel_poses"] = torch.from_numpy(history_rel).unsqueeze(0).to(
            device=control_device,
            dtype=torch.float32,
        )
        tokenized["history_age_steps"] = ages.to(device=control_device)
        return tokenized

    def plan_panoramic(self, payload: dict[str, Any], blobs) -> dict[str, Any]:
        if payload.get("oracle_system2") is not None:
            raise ValueError("Heatmap-control evaluator refuses oracle System2 input")
        if list(payload.get("vlm_image_size") or []) != [384, 384]:
            raise ValueError(f"Native InternNav requires vlm_image_size=[384,384], got {payload.get('vlm_image_size')}")
        if list(payload.get("traj_image_size") or []) != [224, 224]:
            raise ValueError(f"Native InternNav requires traj_image_size=[224,224], got {payload.get('traj_image_size')}")
        expected_options = {
            "system1_coord_order": "generated",
            "trajectory_selection": "mean",
            "trajectory_x_sign": 1.0,
            "trajectory_heading_alignment": "none",
        }
        mismatches = {
            key: {"expected": expected, "actual": payload.get(key)}
            for key, expected in expected_options.items()
            if payload.get(key) != expected
        }
        if mismatches:
            raise ValueError(f"Native evaluation option mismatch: {mismatches}")

        sampling = validate_rpc_sampling_metadata(
            payload.get(HEATMAPVLN_RPC_SAMPLING_FIELD),
            require_deterministic=(
                self.require_deterministic_sampling
                or bool(payload.get("require_deterministic_sampling", False))
            ),
        )
        if sampling is None:
            raise ValueError("Native full evaluation requires deterministic sampling metadata")

        num_history = int(payload.get("num_history", 0))
        if not 0 <= num_history <= 8:
            raise ValueError(f"num_history must be in [0,8], got {num_history}")
        metadata = validate_history_metadata(payload, num_history)
        blob_map = _blobs_by_name(blobs)
        required = {f"current/{view}" for view in VIEW_ORDER} | {"lookdown"}
        required.update(
            f"history/{index}/{view}"
            for index in range(num_history)
            for view in VIEW_ORDER
        )
        missing = sorted(required - set(blob_map))
        if missing:
            raise ValueError(f"Missing heatmap-control RPC image blobs: {missing}")

        current_views = {
            view: _pil_from_blob(blob_map[f"current/{view}"], (384, 384))
            for view in VIEW_ORDER
        }
        history_panoramas = [
            {
                view: _pil_from_blob(blob_map[f"history/{index}/{view}"], (384, 384))
                for view in VIEW_ORDER
            }
            for index in range(num_history)
        ]
        current_front = current_views["front"]
        history_front = [panorama["front"] for panorama in history_panoramas]
        native_lookdown = _pil_from_blob(blob_map["lookdown"])
        if native_lookdown.size != (640, 480):
            raise ValueError(
                "Native InternNav requires a 640x480 conversational lookdown blob, "
                f"got {native_lookdown.size}"
            )
        lookdown = native_lookdown.resize((224, 224))
        instruction = str(payload.get("instruction") or "")
        if not instruction.strip():
            raise ValueError("Native evaluation instruction is empty")

        messages, input_images = build_native_messages(
            instruction,
            history_front,
            current_front,
        )
        output_ids, llm_output, inputs = self._generate(messages, input_images)
        first_output = llm_output
        lookdown_turns = 0
        first_actions = parse_native_actions(llm_output)
        if not re.search(r"\d", llm_output or "") and first_actions[:1] == [ActionCode.LOOKDOWN]:
            messages, input_images = append_native_lookdown_turn(
                messages,
                input_images,
                llm_output,
                native_lookdown,
            )
            output_ids, llm_output, inputs = self._generate(messages, input_images)
            lookdown_turns = 1

        response: dict[str, Any] = {
            "ok": True,
            "proto_v": PROTO_VERSION,
            "control_proto_v": CONTROL_PROTO_VERSION,
            "llm_output": llm_output,
            "native_first_output": first_output,
            "native_lookdown_turns": lookdown_turns,
            "native_front_only": True,
            "native_checkpoint_only": False,
            "native_system2_frozen": True,
            "native_system1_frozen": True,
            "control_mode": self.control_mode,
            "frozen_heatmap_checkpoint_sha256": self.heatmap_dependency[
                "checkpoint_sha256"
            ],
            "control_ema_checkpoint_sha256": self.control_dependency[
                "checkpoint_sha256"
            ],
            "native_model_manifest_sha256": str(
                self.args.native_model_manifest_sha256
            ),
            "control_history_count": num_history,
            "control_history_age_steps": metadata["history_age_steps"].tolist(),
            "pose_provider": metadata["pose_provider"],
            "pose_provider_explicit": metadata["pose_provider_explicit"],
            "pose_ready": metadata["pose_ready"],
            "control_applied": False,
            "control_token_count": 0,
            "system2_source": "internnav_native",
            "oracle_system2": None,
            "pano_goal_view": "front",
            "actions": [],
            "terminal": False,
            "kind": "unknown",
            "candidate_export_proto_v": CANDIDATE_EXPORT_PROTO_VERSION,
            "deployment_arm": self.deployment_arm,
            HEATMAPVLN_RPC_SAMPLING_FIELD: sampling,
        }

        if re.search(r"\d", llm_output or ""):
            coordinates = [int(value) for value in re.findall(r"\d+", llm_output)]
            if len(coordinates) < 2:
                raise RuntimeError(f"Native coordinate output has fewer than two integers: {llm_output!r}")
            pixel_goal = [int(coordinates[1]), int(coordinates[0])]
            response["pixel_goal"] = pixel_goal

            image_grid_thw = inputs.image_grid_thw
            control = self._build_heatmap_control(
                current_front,
                history_front,
                metadata,
                int(sampling["per_call_seed"]),
            )
            response["control_applied"] = control is not None
            response["control_token_count"] = (
                int(control["token_mask"].sum().item()) if control is not None else 0
            )
            if control is None:
                response["control_skip_reason"] = (
                    "pose_not_ready"
                    if history_front and not metadata["pose_ready"]
                    else "no_history"
                )
            with torch.no_grad():
                latent_queries = self.model.latent_queries.expand(1, -1, -1).to(
                    device=self.device,
                    dtype=self.model.config.dtype,
                )
                trajectory_latents = self.model.qwen2_5_vl.generate_latents(
                    output_ids=output_ids,
                    pixel_values=inputs.pixel_values,
                    image_grid_thw=image_grid_thw,
                    latent_queries=latent_queries,
                    attention_mask=inputs.get("attention_mask"),
                    mm_token_type_ids=inputs.get("mm_token_type_ids"),
                )
                trajectory_latents = self.model.adapt_traj_hidden_states(
                    trajectory_latents
                )
                image_dp = torch.from_numpy(np.array(lookdown)).to(
                    device=self.device,
                    dtype=torch.bfloat16,
                ) / 255.0
                trajectory_images = torch.stack([image_dp.clone(), image_dp]).unsqueeze(0)
                generator = torch.Generator(device=self.device)
                generator.manual_seed(int(sampling["per_call_seed"]))
                action_head = self.model.nextdit_action_head
                noise_shape = (
                    int(trajectory_latents.shape[0]) * int(action_head.config.num_sample_trajs),
                    int(action_head.config.predict_steps),
                    int(action_head.config.action_dim),
                )
                initial_noise = torch.randn(
                    noise_shape,
                    generator=generator,
                    device=self.device,
                    dtype=trajectory_latents.dtype,
                )
                native_trajectory = action_head.get_trajectory(
                    trajectory_latents,
                    traj_images=trajectory_images,
                    initial_noise=initial_noise,
                )
                heatmap_trajectory = action_head.get_trajectory(
                    trajectory_latents,
                    traj_images=trajectory_images,
                    heatmap_tokens=(control["tokens"] if control is not None else None),
                    heatmap_mask=(control["token_mask"] if control is not None else None),
                    heatmap_valid=(control["sample_valid"] if control is not None else None),
                    initial_noise=initial_noise,
                )

            if native_trajectory.shape != heatmap_trajectory.shape:
                raise RuntimeError(
                    "Paired candidate shapes differ: "
                    f"native={tuple(native_trajectory.shape)} "
                    f"heatmap={tuple(heatmap_trajectory.shape)}"
                )
            if tuple(native_trajectory.shape) != noise_shape:
                raise RuntimeError(
                    f"Unexpected candidate shape {tuple(native_trajectory.shape)}; "
                    f"expected {noise_shape}"
                )
            if control is None and not torch.equal(native_trajectory, heatmap_trajectory):
                raise RuntimeError(
                    "Control-absent paired arms must be bit-identical"
                )
            paired_delta = (
                heatmap_trajectory.detach().float()
                - native_trajectory.detach().float()
            )

            compact_arrays: dict[str, Any] = {
                "native_trajectories": _tensor_float32(native_trajectory),
                "heatmap_trajectories": _tensor_float32(heatmap_trajectory),
                "initial_noise_bf16_bits": _tensor_bfloat16_bits(initial_noise),
                "system2_latent_bf16_bits": _tensor_bfloat16_bits(trajectory_latents),
                "system2_input_ids": np.ascontiguousarray(
                    inputs.input_ids.detach().cpu().numpy().astype(np.int32)
                ),
                "system2_output_ids": np.ascontiguousarray(
                    output_ids.detach().cpu().numpy().astype(np.int32)
                ),
                "pixel_goal": np.asarray(pixel_goal, dtype=np.int32),
                "history_capture_steps": np.asarray(
                    metadata["history_capture_steps"], dtype=np.int32
                ),
                "history_age_steps": np.asarray(
                    metadata["history_age_steps"], dtype=np.int32
                ),
            }
            if metadata["pose_provider"] == GT_POSE_PROVIDER:
                compact_arrays.update(
                    {
                        "current_c2w": np.asarray(
                            metadata["current_c2w"], dtype=np.float32
                        ),
                        "history_c2w": np.asarray(
                            metadata["history_c2w"], dtype=np.float32
                        ),
                    }
                )
            else:
                compact_arrays.update(
                    {
                        "provided_history_rel_poses": np.asarray(
                            metadata["history_rel_poses"], dtype=np.float32
                        ),
                        "vo_current_frame_id": np.asarray(
                            metadata["vo_current_frame_id"], dtype=np.int64
                        ),
                        "vo_history_frame_ids": np.asarray(
                            metadata["vo_history_frame_ids"], dtype=np.int64
                        ),
                        "vo_trajectory_revision": np.asarray(
                            metadata["vo_trajectory_revision"], dtype=np.int64
                        ),
                    }
                )
            if control is None:
                compact_arrays.update(
                    {
                        "heatmap_tokens": np.empty((1, 0, 128), dtype=np.float16),
                        "heatmap_token_mask": np.empty((1, 0), dtype=np.bool_),
                        "heatmap_sample_valid": np.zeros((1,), dtype=np.bool_),
                    }
                )
            else:
                float16_control_fields = {
                    "heatmap_tokens": "tokens",
                    "heatmap_logits": "heatmap_logits",
                    "visibility_logits": "visibility_logits",
                    "coarse_probabilities": "coarse_probabilities",
                    "spatial_statistics": "spatial_statistics",
                    "view_probabilities": "view_probabilities",
                    "none_probability": "none_probability",
                    "normalized_age": "normalized_age",
                    "history_rank": "history_rank",
                    "structured_features": "structured_features",
                }
                for array_name, field_name in float16_control_fields.items():
                    compact_arrays[array_name] = _tensor_float16(control[field_name])
                compact_arrays.update(
                    {
                        "heatmap_token_mask": np.ascontiguousarray(
                            control["token_mask"].detach().cpu().numpy().astype(np.bool_)
                        ),
                        "heatmap_sample_valid": np.ascontiguousarray(
                            control["sample_valid"].detach().cpu().numpy().astype(np.bool_)
                        ),
                        "fixed_history_mask": np.ascontiguousarray(
                            control["history_mask"].detach().cpu().numpy().astype(np.bool_)
                        ),
                        "fixed_history_rel_poses": _tensor_float32(
                            control["history_rel_poses"]
                        ),
                        "fixed_history_age_steps": np.ascontiguousarray(
                            control["history_age_steps"].detach().cpu().numpy().astype(np.int32)
                        ),
                    }
                )

            candidate_blob, candidate_manifest = _pack_compact_arrays(compact_arrays)
            scheduler_config = _json_safe(dict(action_head.noise_scheduler.config))
            candidate_manifest["compact_feature_schema"] = candidate_manifest["schema"]
            candidate_manifest.update(
                {
                    "schema": AUDIT_SCHEMA_VERSION,
                    "export_proto_v": CANDIDATE_EXPORT_PROTO_VERSION,
                    "paired_explicit_initial_noise": True,
                    "initial_noise_array": "initial_noise_bf16_bits",
                    "initial_noise_logical_dtype": "torch.bfloat16",
                    "initial_noise_sha256": candidate_manifest["arrays"][
                        "initial_noise_bf16_bits"
                    ]["sha256"],
                    "candidate_count_per_arm": int(action_head.config.num_sample_trajs),
                    "predict_steps": int(action_head.config.predict_steps),
                    "action_dim": int(action_head.config.action_dim),
                    "num_inference_steps": int(action_head.config.num_inference_steps),
                    "guidance_scale": float(action_head.config.guidance_scale),
                    "scheduler_class": type(action_head.noise_scheduler).__name__,
                    "scheduler_config": scheduler_config,
                    "server_source_sha256": _file_sha256(__file__),
                    "source_sha256": dict(self.candidate_source_sha256),
                    "evaluation_config_sha256": _file_sha256(self.args.config),
                    "native_model_manifest_sha256": str(
                        self.args.native_model_manifest_sha256
                    ),
                    "frozen_heatmap_checkpoint_sha256": self.heatmap_dependency[
                        "checkpoint_sha256"
                    ],
                    "control_ema_checkpoint_sha256": self.control_dependency[
                        "checkpoint_sha256"
                    ],
                    "pose_provider": metadata["pose_provider"],
                    "pose_provider_explicit": metadata["pose_provider_explicit"],
                    "pose_ready": metadata["pose_ready"],
                    "control_applied": control is not None,
                    "paired_delta_l2": float(paired_delta.norm().item()),
                    "paired_delta_max_abs": float(paired_delta.abs().max().item()),
                    "instruction_sha256": sha256_bytes(instruction.encode("utf-8")),
                    "request_blob_sha256": {
                        name: sha256_bytes(bytes(blob.data))
                        for name, blob in sorted(blob_map.items())
                    },
                }
            )
            if metadata["pose_provider"] == AMB3R_POSE_PROVIDER:
                candidate_manifest["vo_pose_metadata"] = {
                    "current_frame_id": int(metadata["vo_current_frame_id"]),
                    "history_frame_ids": metadata[
                        "vo_history_frame_ids"
                    ].tolist(),
                    "provider_phase": metadata["vo_provider_phase"],
                    "trajectory_revision": int(
                        metadata["vo_trajectory_revision"]
                    ),
                }

            from internnav.model.utils.vln_utils import traj_to_actions

            deployment_trajectory = (
                native_trajectory
                if self.deployment_arm == "native"
                else heatmap_trajectory
            )
            actions = _finalize_local_actions(
                traj_to_actions(deployment_trajectory.detach().clone())
            )
            if actions and actions[0] == ActionCode.STOP:
                actions = [ActionCode.LEFT]
                response["anti_deadlock"] = True
            response.update(
                {
                    "kind": "trajectory",
                    "actions": actions,
                    "trajectory_summary": _trajectory_summary(deployment_trajectory),
                    "native_trajectory_summary": _trajectory_summary(native_trajectory),
                    "heatmap_trajectory_summary": _trajectory_summary(heatmap_trajectory),
                    "trajectory_x_sign": 1.0,
                    "trajectory_heading_alignment": "none",
                    "candidate_audit": candidate_manifest,
                    "_candidate_audit_blob": candidate_blob,
                }
            )
            return response

        actions = parse_native_actions(llm_output)
        if actions:
            if actions[0] == ActionCode.STOP:
                response.update(
                    {
                        "kind": "stop",
                        "terminal": True,
                        "actions": [ActionCode.STOP],
                    }
                )
            else:
                response.update({"kind": "native_actions", "actions": actions})
            return response

        response.update(
            {
                "kind": "fallback_stop",
                "terminal": True,
                "actions": [ActionCode.STOP],
            }
        )
        return response


class HeatmapControlServicer(vla_pb2_grpc.VLAServicer):
    def __init__(self, runtime: HeatmapControlRuntime):
        self.runtime = runtime
        self.requests_processed = 0

    def InferJSON(self, request: vla_pb2.JSONRequest, context) -> vla_pb2.JSONResponse:
        try:
            payload = json.loads(request.json_payload) if request.json_payload else {}
            if request.method != "plan_panoramic":
                raise ValueError(f"Unsupported method: {request.method}")
            output = self.runtime.plan_panoramic(payload, request.blobs)
            candidate_blob = output.pop("_candidate_audit_blob", None)
            response_blobs = []
            if candidate_blob is not None:
                response_blobs.append(
                    vla_pb2.BinaryBlob(
                        name=CANDIDATE_BLOB_NAME,
                        data=candidate_blob,
                        mime_type="application/x-npz",
                        meta_json=json.dumps(
                            output.get("candidate_audit", {}),
                            ensure_ascii=False,
                            sort_keys=True,
                        ),
                    )
                )
            self.requests_processed += 1
            return vla_pb2.JSONResponse(
                ts=request.ts,
                json_payload=json.dumps(output, ensure_ascii=False),
                blobs=response_blobs,
                model_v=(
                    "candidate-support-r2r-paired:"
                    f"deployment-{self.runtime.deployment_arm}"
                ),
            )
        except Exception as exc:
            LOGGER.exception("InferJSON failed")
            context.set_details(str(exc))
            context.set_code(grpc.StatusCode.INTERNAL)
            return vla_pb2.JSONResponse(
                ts=request.ts,
                json_payload=json.dumps({"ok": False, "error": str(exc)}),
            )

    def HealthCheck(self, request, context) -> vla_pb2.HealthCheckResponse:
        return vla_pb2.HealthCheckResponse(
            status=vla_pb2.HealthCheckResponse.SERVING,
            message="Paired candidate-support InternNav model server is running",
            version=PROTO_VERSION,
            requests_processed=self.requests_processed,
        )

    def GetServerInfo(self, request, context) -> vla_pb2.ServerInfo:
        return vla_pb2.ServerInfo(
            version=PROTO_VERSION,
            model_version=(
                "candidate-support-r2r-paired:"
                f"deployment-{self.runtime.deployment_arm}"
            ),
            max_batch_size=1,
            supported_formats=["json+jpeg", CANDIDATE_EXPORT_PROTO_VERSION],
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Paired native/control candidate-support RPC exporter"
    )
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", required=True)
    parser.add_argument("--frozen_heatmap_checkpoint", required=True)
    parser.add_argument("--frozen_heatmap_sha256", required=True)
    parser.add_argument("--control_checkpoint", required=True)
    parser.add_argument("--control_checkpoint_sha256", required=True)
    parser.add_argument("--native_model_manifest", required=True)
    parser.add_argument("--native_model_manifest_sha256", required=True)
    parser.add_argument(
        "--control_mode",
        choices=("on",),
        default="on",
    )
    parser.add_argument(
        "--deployment_arm",
        choices=("native", "heatmap_control"),
        default="native",
        help="Arm executed in Habitat while both paired arms are exported.",
    )
    parser.add_argument("--gpu_id", type=int, default=0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=51500)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--require_deterministic_sampling", action="store_true")
    parser.add_argument("--log_level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    runtime = HeatmapControlRuntime(args)
    server = grpc.server(
        futures.ThreadPoolExecutor(max_workers=args.workers),
        options=[
            ("grpc.max_send_message_length", 128 * 1024 * 1024),
            ("grpc.max_receive_message_length", 128 * 1024 * 1024),
        ],
    )
    vla_pb2_grpc.add_VLAServicer_to_server(HeatmapControlServicer(runtime), server)
    address = f"{args.host}:{args.port}"
    if server.add_insecure_port(address) == 0:
        raise RuntimeError(f"Could not bind candidate-support RPC server to {address}")
    server.start()
    LOGGER.info("Candidate-support RPC server listening on %s", address)

    def shutdown(_signum, _frame):
        LOGGER.info("Stopping candidate-support RPC server")
        server.stop(grace=5)
        raise SystemExit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)
    server.wait_for_termination()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

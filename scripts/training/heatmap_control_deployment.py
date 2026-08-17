"""Fail-closed deployment checkpoint validation for heatmap control."""

from __future__ import annotations

import hashlib
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch

from .native_internnav_dependency import (
    NativeInternNavDependencyError,
    RUNTIME_KEY as NATIVE_DEPENDENCY_RUNTIME_KEY,
    validate_native_internnav_dependency_contract,
)


_SHA256_RE = re.compile(r"[0-9a-f]{64}")
_CONTROL_KEY_RE = re.compile(
    r"nextdit_action_head\.traj_dit\.model\.layers\."
    r"(?P<layer>[0-9]+)\.heatmap_control\."
)


class HeatmapControlDeploymentError(RuntimeError):
    """The checkpoint is not the requested complete EMA deployment state."""


def _mapping(value: Any, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HeatmapControlDeploymentError(f"{name} must be a mapping")
    return value


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_weights_only(path: Path) -> Mapping[str, Any]:
    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise HeatmapControlDeploymentError(
            "runtime lacks mandatory torch.load(weights_only=True)"
        ) from exc
    except Exception as exc:
        raise HeatmapControlDeploymentError(
            f"unable to load deployment checkpoint safely: {path}"
        ) from exc
    return _mapping(payload, "deployment checkpoint")


def _normalized_ema_shadow(value: Any) -> dict[str, torch.Tensor]:
    shadow = _mapping(value, "ema_state_dict.shadow")
    result: dict[str, torch.Tensor] = {}
    for raw_name, tensor in shadow.items():
        if not isinstance(raw_name, str) or not raw_name:
            raise HeatmapControlDeploymentError(
                "ema_state_dict.shadow contains an invalid key"
            )
        name = raw_name
        while name.startswith("module."):
            name = name[len("module.") :]
        name = name.replace(".module.", ".")
        if name in result:
            raise HeatmapControlDeploymentError(
                f"ema_state_dict.shadow has duplicate normalized key: {name}"
            )
        if not torch.is_tensor(tensor):
            raise HeatmapControlDeploymentError(
                f"ema_state_dict.shadow value is not a tensor: {name}"
            )
        result[name] = tensor
    return result


def validate_heatmap_control_deployment_checkpoint(
    checkpoint_path: str | Path,
    *,
    expected_heatmap_sha256: str,
    expected_policy_fingerprint: str,
    expected_collection_roots: Sequence[str],
    expected_epoch: int = 3,
) -> dict[str, Any]:
    """Validate and fingerprint the exact full-epoch EMA eval artifact."""

    path = Path(checkpoint_path).resolve(strict=True)
    if not path.is_file() or path.is_symlink():
        raise HeatmapControlDeploymentError(
            f"deployment checkpoint must be a regular non-symlink file: {path}"
        )
    if _SHA256_RE.fullmatch(expected_heatmap_sha256) is None:
        raise HeatmapControlDeploymentError("expected heatmap SHA-256 is malformed")
    roots = list(expected_collection_roots)
    if len(roots) != 4 or len(set(roots)) != 4 or any(not root for root in roots):
        raise HeatmapControlDeploymentError(
            "deployment contract requires four ordered, unique DAgger roots"
        )

    actual_sha256 = _sha256(path)
    payload = _load_weights_only(path)
    if type(payload.get("epoch")) is not int or payload["epoch"] != expected_epoch:
        raise HeatmapControlDeploymentError(
            f"deployment checkpoint must be epoch={expected_epoch}"
        )
    if type(payload.get("stage_idx")) is not int or payload["stage_idx"] != 0:
        raise HeatmapControlDeploymentError(
            "deployment checkpoint must have stage_idx=0"
        )
    if payload.get("batch") is not None:
        raise HeatmapControlDeploymentError(
            "deployment checkpoint must be a complete epoch checkpoint"
        )
    if payload.get("stage_name") != "heatmap_system1_control":
        raise HeatmapControlDeploymentError(
            "deployment checkpoint has the wrong stage_name"
        )
    semantics = _mapping(payload.get("weight_semantics"), "weight_semantics")
    if semantics.get("trainable_state_dict") != "ema":
        raise HeatmapControlDeploymentError(
            "deployment trainable_state_dict must contain EMA weights"
        )

    state = _mapping(payload.get("trainable_state_dict"), "trainable_state_dict")
    if not state:
        raise HeatmapControlDeploymentError("deployment state is empty")
    tokenizer_tensors = 0
    control_layers: set[int] = set()
    for name, value in state.items():
        if not isinstance(name, str) or not name:
            raise HeatmapControlDeploymentError("deployment state has an invalid key")
        if name.startswith("heatmap_tokenizer."):
            tokenizer_tensors += 1
        else:
            match = _CONTROL_KEY_RE.match(name)
            if match is None:
                raise HeatmapControlDeploymentError(
                    f"forbidden deployment tensor outside control modules: {name}"
                )
            control_layers.add(int(match.group("layer")))
        if not torch.is_tensor(value) or value.layout != torch.strided:
            raise HeatmapControlDeploymentError(
                f"deployment value is not a dense tensor: {name}"
            )
        if not value.is_floating_point():
            raise HeatmapControlDeploymentError(
                f"deployment tensor is not floating point: {name}"
            )
        if not torch.isfinite(value).all().item():
            raise HeatmapControlDeploymentError(
                f"deployment tensor is non-finite: {name}"
            )
    if tokenizer_tensors < 1:
        raise HeatmapControlDeploymentError(
            "deployment state contains no heatmap tokenizer tensors"
        )
    if control_layers != set(range(12)):
        raise HeatmapControlDeploymentError(
            "deployment state must contain heatmap control for layers 0..11; "
            f"got {sorted(control_layers)}"
        )

    ema_state = _mapping(payload.get("ema_state_dict"), "ema_state_dict")
    ema_shadow = _normalized_ema_shadow(ema_state.get("shadow"))
    if set(ema_shadow) != set(state):
        raise HeatmapControlDeploymentError(
            "deployment trainable_state_dict keys do not exactly match the "
            "normalized EMA shadow"
        )
    for name, deployment_tensor in state.items():
        ema_tensor = ema_shadow[name]
        if (
            tuple(deployment_tensor.shape) != tuple(ema_tensor.shape)
            or deployment_tensor.dtype != ema_tensor.dtype
            or not torch.equal(deployment_tensor, ema_tensor)
        ):
            raise HeatmapControlDeploymentError(
                "deployment trainable_state_dict is not an exact EMA tensor "
                f"for {name}"
            )

    config = _mapping(payload.get("config"), "checkpoint config")
    model = _mapping(config.get("model"), "checkpoint config.model")
    llm = _mapping(model.get("llm"), "checkpoint config.model.llm")
    action = _mapping(model.get("action_head"), "checkpoint config.model.action_head")
    nextdit = _mapping(action.get("nextdit"), "checkpoint config nextdit")
    control = _mapping(nextdit.get("heatmap_control"), "checkpoint config control")
    if control.get("heatmap_checkpoint_sha256") != expected_heatmap_sha256:
        raise HeatmapControlDeploymentError(
            "checkpoint config frozen heatmap SHA-256 mismatch"
        )
    runtime = _mapping(config.get("runtime"), "checkpoint config.runtime")
    try:
        native_dependency = validate_native_internnav_dependency_contract(
            runtime.get(NATIVE_DEPENDENCY_RUNTIME_KEY),
            expected_model_path=llm.get("model_path"),
            name=f"checkpoint config.runtime.{NATIVE_DEPENDENCY_RUNTIME_KEY}",
        )
    except NativeInternNavDependencyError as exc:
        raise HeatmapControlDeploymentError(str(exc)) from exc
    if nextdit.get("internnav_model_path") != native_dependency["model_path"]:
        raise HeatmapControlDeploymentError(
            "checkpoint NextDiT path differs from the locked native dependency"
        )
    dependency = _mapping(
        runtime.get("frozen_heatmap_dependency"),
        "checkpoint frozen_heatmap_dependency",
    )
    if dependency.get("checkpoint_sha256") != expected_heatmap_sha256:
        raise HeatmapControlDeploymentError(
            "checkpoint runtime frozen heatmap SHA-256 mismatch"
        )

    data = _mapping(config.get("data"), "checkpoint config.data")
    mixture = _mapping(data.get("mixture"), "checkpoint config.data.mixture")
    if (
        data.get("dataset_type") != "expert_dagger_mixture"
        or data.get("in_order") is not True
        or mixture.get("profile") != "expert50_normal20_hard30"
        or mixture.get("seed") != 42
        or mixture.get("epoch_size") != 72000
    ):
        raise HeatmapControlDeploymentError(
            "checkpoint config does not match the locked 50/20/30 72k mixture"
        )
    dagger = _mapping(
        data.get("trajectory_dagger"),
        "checkpoint config.data.trajectory_dagger",
    )
    if list(dagger.get("collection_roots") or ()) != roots:
        raise HeatmapControlDeploymentError(
            "checkpoint DAgger roots differ from the sealed launcher roots"
        )
    if dagger.get("expected_policy_fingerprint") != expected_policy_fingerprint:
        raise HeatmapControlDeploymentError(
            "checkpoint native policy fingerprint mismatch"
        )

    return {
        "schema": "heatmap-control-deployment-validation-v1",
        "checkpoint_path": str(path),
        "checkpoint_sha256": actual_sha256,
        "epoch": expected_epoch,
        "tensor_count": len(state),
        "tokenizer_tensor_count": tokenizer_tensors,
        "control_layers": sorted(control_layers),
        "native_internnav_dependency": native_dependency,
    }


__all__ = [
    "HeatmapControlDeploymentError",
    "validate_heatmap_control_deployment_checkpoint",
]

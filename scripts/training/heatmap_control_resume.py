"""Fail-closed resume boundary for frozen-native heatmap control training.

Heatmap-control checkpoints are deliberately delta-only: they may restore the
structured heatmap tokenizer and per-NextDiT-layer control branches, but they
must never carry Qwen, native InternNav System1, LoRA, or generic warm-start
weights.  Validation happens before the generic loader mutates the model.
"""

from __future__ import annotations

import os
import re
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn

from .frozen_heatmap_checkpoint import (
    DEPENDENCY_SCHEMA,
    STATE_KEY as FROZEN_HEATMAP_STATE_KEY,
    TARGET_PREFIX as FROZEN_HEATMAP_PREFIX,
    compute_file_sha256,
)
from .native_internnav_dependency import (
    NativeInternNavDependencyError,
    RUNTIME_KEY as NATIVE_DEPENDENCY_RUNTIME_KEY,
    validate_native_internnav_dependency_contract,
)


TOKENIZER_PREFIX = "heatmap_tokenizer."
CONTROL_LAYER_PREFIX = "nextdit_action_head.traj_dit.model.layers."
CONTROL_MARKER = ".heatmap_control."
_SHA256_PATTERN = re.compile(r"[0-9a-f]{64}")
_CONTROL_ARCH_DEFAULTS = {
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
_FORBIDDEN_STAGE_FLAGS = (
    "requires_base_checkpoint",
    "bridge_only",
    "base_checkpoint_lora_only",
    "deployment_include_frozen_lora",
    "merge_frozen_lora",
)


class HeatmapControlResumeError(RuntimeError):
    """A checkpoint could escape the heatmap-control-only resume boundary."""


def heatmap_control_enabled(cfg: Mapping[str, Any]) -> bool:
    """Return whether the strict heatmap-control path is enabled."""
    try:
        control = cfg["model"]["action_head"]["nextdit"]["heatmap_control"]
    except (KeyError, TypeError):
        return False
    return isinstance(control, Mapping) and bool(control.get("enabled", False))


def reject_heatmap_control_load_weights(
    cfg: Mapping[str, Any],
    load_weights: str | os.PathLike[str] | None,
) -> None:
    """Forbid generic initialization checkpoints for heatmap-control runs."""
    if heatmap_control_enabled(cfg) and load_weights:
        raise HeatmapControlResumeError(
            "heatmap control forbids --load-weights: new training must start "
            "from the released native InternNav model plus the independently "
            "verified frozen heatmap dependency; use --resume only for an "
            "existing heatmap-control run"
        )


def _canonical_key(raw_name: Any) -> str:
    if not isinstance(raw_name, str) or not raw_name:
        raise HeatmapControlResumeError(
            "checkpoint state keys must be non-empty strings"
        )
    name = raw_name
    previous = None
    while name != previous:
        previous = name
        while name.startswith("module."):
            name = name[len("module.") :]
        name = name.replace(".module.", ".")
    if not name:
        raise HeatmapControlResumeError(
            f"checkpoint key normalizes to an empty name: {raw_name!r}"
        )
    return name


def _is_control_parameter(name: str) -> bool:
    return name.startswith(TOKENIZER_PREFIX) or (
        name.startswith(CONTROL_LAYER_PREFIX) and CONTROL_MARKER in name
    )


def _expected_control_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
    if not isinstance(model, nn.Module):
        raise HeatmapControlResumeError("model must be a torch.nn.Module")
    expected: dict[str, nn.Parameter] = {}
    raw_for_canonical: dict[str, str] = {}
    for raw_name, parameter in model.named_parameters():
        name = _canonical_key(raw_name)
        if not _is_control_parameter(name):
            continue
        if name in expected:
            raise HeatmapControlResumeError(
                "model parameter names collide after module normalization: "
                f"{raw_for_canonical[name]!r} and {raw_name!r} -> {name!r}"
            )
        expected[name] = parameter
        raw_for_canonical[name] = raw_name

    tokenizer_names = [name for name in expected if name.startswith(TOKENIZER_PREFIX)]
    control_names = [name for name in expected if CONTROL_MARKER in name]
    if not tokenizer_names or not control_names:
        raise HeatmapControlResumeError(
            "constructed model does not contain both heatmap_tokenizer.* and "
            "per-layer *.heatmap_control.* parameters"
        )
    return expected


def _validate_tensor_state(
    raw_state: Any,
    expected: Mapping[str, nn.Parameter],
    *,
    state_name: str,
) -> dict[str, torch.Tensor]:
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise HeatmapControlResumeError(
            f"{state_name} must be a non-empty tensor mapping"
        )

    normalized: dict[str, torch.Tensor] = {}
    raw_for_canonical: dict[str, str] = {}
    for raw_name, value in raw_state.items():
        name = _canonical_key(raw_name)
        if name in normalized:
            raise HeatmapControlResumeError(
                f"{state_name} has duplicate keys after module normalization: "
                f"{raw_for_canonical[name]!r} and {raw_name!r} -> {name!r}"
            )
        if not torch.is_tensor(value):
            raise HeatmapControlResumeError(
                f"{state_name}[{name!r}] is not a tensor"
            )
        if value.layout != torch.strided:
            raise HeatmapControlResumeError(
                f"{state_name}[{name!r}] must be a dense strided tensor"
            )
        normalized[name] = value
        raw_for_canonical[name] = raw_name

    actual_names = set(normalized)
    expected_names = set(expected)
    missing = sorted(expected_names - actual_names)
    unexpected = sorted(actual_names - expected_names)
    if missing or unexpected:
        raise HeatmapControlResumeError(
            f"{state_name} must exactly cover the heatmap-control delta: "
            f"missing={missing[:8]}, unexpected={unexpected[:8]}"
        )

    for name in sorted(expected_names):
        value = normalized[name]
        target = expected[name]
        if tuple(value.shape) != tuple(target.shape):
            raise HeatmapControlResumeError(
                f"{state_name} shape mismatch for {name}: "
                f"checkpoint={tuple(value.shape)}, model={tuple(target.shape)}"
            )
        if (value.is_floating_point() or value.is_complex()) and not bool(
            torch.isfinite(value).all().item()
        ):
            raise HeatmapControlResumeError(
                f"{state_name} contains non-finite values: {name}"
            )
    return normalized


def _require_mapping(value: Any, *, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HeatmapControlResumeError(f"{name} must be a mapping")
    return value


def _requires_grad_scaler(cfg: Mapping[str, Any], *, name: str) -> bool:
    optim = cfg.get("optim") or {}
    if not isinstance(optim, Mapping):
        raise HeatmapControlResumeError(f"{name}.optim must be a mapping")
    amp = str(optim.get("amp", "bf16")).strip().lower()
    return amp in {"fp16", "float16", "torch.float16"}


def _validate_training_runtime_states(
    checkpoint: Mapping[str, Any],
    current_cfg: Mapping[str, Any],
    saved_cfg: Mapping[str, Any],
) -> bool:
    """Require the optimizer-matched state needed for an exact resume."""
    optimizer_state = _require_mapping(
        checkpoint.get("optimizer_state_dict"),
        name="optimizer_state_dict",
    )
    param_groups = optimizer_state.get("param_groups")
    state = optimizer_state.get("state")
    if not isinstance(param_groups, list) or not param_groups:
        raise HeatmapControlResumeError(
            "optimizer_state_dict.param_groups must be a non-empty list"
        )
    if not isinstance(state, Mapping):
        raise HeatmapControlResumeError(
            "optimizer_state_dict.state must be a mapping"
        )

    scheduler_state = _require_mapping(
        checkpoint.get("scheduler_state_dict"),
        name="scheduler_state_dict",
    )
    if not scheduler_state:
        raise HeatmapControlResumeError(
            "scheduler_state_dict must be non-empty for an exact resume"
        )

    current_requires_scaler = _requires_grad_scaler(
        current_cfg,
        name="current config",
    )
    saved_requires_scaler = _requires_grad_scaler(
        saved_cfg,
        name="saved config",
    )
    if current_requires_scaler != saved_requires_scaler:
        raise HeatmapControlResumeError(
            "saved and current AMP modes disagree about GradScaler state"
        )
    if current_requires_scaler:
        scaler_state = _require_mapping(
            checkpoint.get("scaler_state_dict"),
            name="scaler_state_dict",
        )
        if not scaler_state:
            raise HeatmapControlResumeError(
                "fp16 heatmap-control resume requires a non-empty scaler_state_dict"
            )
    return current_requires_scaler


def _control_config(cfg: Mapping[str, Any], *, name: str) -> Mapping[str, Any]:
    try:
        model_cfg = _require_mapping(cfg["model"], name=f"{name}.model")
        action_cfg = _require_mapping(
            model_cfg["action_head"], name=f"{name}.model.action_head"
        )
        nextdit_cfg = _require_mapping(
            action_cfg["nextdit"], name=f"{name}.model.action_head.nextdit"
        )
        control_cfg = _require_mapping(
            nextdit_cfg["heatmap_control"],
            name=f"{name}.model.action_head.nextdit.heatmap_control",
        )
    except KeyError as exc:
        raise HeatmapControlResumeError(
            f"{name} is missing heatmap-control model configuration: {exc}"
        ) from exc
    if not bool(action_cfg.get("enable", False)):
        raise HeatmapControlResumeError(f"{name} disables the native action head")
    if not bool(nextdit_cfg.get("enabled", False)):
        raise HeatmapControlResumeError(f"{name} disables native NextDiT System1")
    if not bool(control_cfg.get("enabled", False)):
        raise HeatmapControlResumeError(f"{name} disables heatmap control")
    return control_cfg


def _normalized_model_path(value: Any, *, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HeatmapControlResumeError(f"{name} must be a non-empty path")
    return os.path.normpath(os.path.expanduser(value.strip()))


def _resolve_checkpoint_stage(
    cfg: Mapping[str, Any],
    *,
    stage_idx: int,
    stage_name: str,
    config_name: str,
) -> Mapping[str, Any]:
    training = _require_mapping(cfg.get("training"), name=f"{config_name}.training")
    stages = training.get("stages")
    if not isinstance(stages, list) or not stages:
        raise HeatmapControlResumeError(
            f"{config_name}.training.stages must be a non-empty list"
        )
    candidate = None
    if 0 <= stage_idx < len(stages) and isinstance(stages[stage_idx], Mapping):
        indexed = stages[stage_idx]
        if not stage_name or indexed.get("name") == stage_name:
            candidate = indexed
    if candidate is None and stage_name:
        candidate = next(
            (
                stage
                for stage in stages
                if isinstance(stage, Mapping) and stage.get("name") == stage_name
            ),
            None,
        )
    if candidate is None:
        raise HeatmapControlResumeError(
            f"{config_name} has no stage matching index={stage_idx}, name={stage_name!r}"
        )
    return candidate


def _assert_control_stage(stage: Mapping[str, Any], *, name: str) -> None:
    if list(stage.get("trainable_modules", ())) != [
        "heatmap_tokenizer",
        "heatmap_control",
    ]:
        raise HeatmapControlResumeError(
            f"{name} must train exactly heatmap_tokenizer and heatmap_control"
        )
    if stage.get("strict_trainable_modules") is not True:
        raise HeatmapControlResumeError(
            f"{name} must set strict_trainable_modules=true"
        )
    if not bool(stage.get("train_action", False)):
        raise HeatmapControlResumeError(f"{name} must set train_action=true")
    frozen_flags = (
        bool(stage.get("train_heatmap", False)),
        bool(stage.get("train_history", False)),
        bool(stage.get("train_future", False)),
        bool(stage.get("train_lm", stage.get("train_system2_sft", False))),
        bool(stage.get("train_system2_sft", False)),
    )
    if any(frozen_flags):
        raise HeatmapControlResumeError(
            f"{name} attempts to train a frozen heatmap/System2 component"
        )
    enabled_forbidden = [flag for flag in _FORBIDDEN_STAGE_FLAGS if stage.get(flag)]
    if enabled_forbidden:
        raise HeatmapControlResumeError(
            f"{name} contains forbidden base/LoRA override flags: {enabled_forbidden}"
        )
    if stage.get("heatmap_warmstart_contract"):
        raise HeatmapControlResumeError(
            f"{name} must not contain a generic heatmap warm-start contract"
        )


def _assert_native_config(
    cfg: Mapping[str, Any],
    *,
    name: str,
    stage_idx: int,
    stage_name: str,
) -> tuple[str, Mapping[str, Any]]:
    control_cfg = _control_config(cfg, name=name)
    model_cfg = _require_mapping(cfg.get("model"), name=f"{name}.model")
    llm_cfg = _require_mapping(model_cfg.get("llm"), name=f"{name}.model.llm")
    nextdit_cfg = _require_mapping(
        _require_mapping(
            model_cfg.get("action_head"), name=f"{name}.model.action_head"
        ).get("nextdit"),
        name=f"{name}.model.action_head.nextdit",
    )

    if bool(llm_cfg.get("use_lora", False)):
        raise HeatmapControlResumeError(f"{name} enables forbidden System2 LoRA")
    llm_path = _normalized_model_path(
        llm_cfg.get("model_path"), name=f"{name}.model.llm.model_path"
    )
    system1_path = _normalized_model_path(
        nextdit_cfg.get("internnav_model_path"),
        name=f"{name}.model.action_head.nextdit.internnav_model_path",
    )
    if llm_path != system1_path:
        raise HeatmapControlResumeError(
            f"{name} does not use one unified original InternNav path"
        )

    override_keys = (
        "internnav_system1_path",
        "pretrained_system1_path",
        "dav2_ckpt_path",
    )
    active_overrides = [
        key for key in override_keys if nextdit_cfg.get(key) not in (None, "")
    ]
    if active_overrides:
        raise HeatmapControlResumeError(
            f"{name} contains forbidden System1 overrides: {active_overrides}"
        )
    if int(nextdit_cfg.get("warmup_steps", 0)) != 0:
        raise HeatmapControlResumeError(
            f"{name} must keep native NextDiT warmup_steps=0"
        )
    pano_adapter = nextdit_cfg.get("pano_latent_adapter") or {}
    if not isinstance(pano_adapter, Mapping):
        raise HeatmapControlResumeError(
            f"{name}.model.action_head.nextdit.pano_latent_adapter must be a mapping"
        )
    if bool(pano_adapter.get("enabled", False)) or pano_adapter.get(
        "pretrained_path"
    ) not in (None, ""):
        raise HeatmapControlResumeError(
            f"{name} contains a forbidden panoramic/System1 adapter override"
        )

    runtime = cfg.get("runtime") or {}
    if not isinstance(runtime, Mapping):
        raise HeatmapControlResumeError(f"{name}.runtime must be a mapping")
    forbidden_runtime = [
        key
        for key in ("base_checkpoint", "single_view_heatmap_init_artifact")
        if runtime.get(key) not in (None, "")
    ]
    if forbidden_runtime:
        raise HeatmapControlResumeError(
            f"{name}.runtime contains forbidden inferred/warm-start dependencies: "
            f"{forbidden_runtime}"
        )

    stage = _resolve_checkpoint_stage(
        cfg,
        stage_idx=stage_idx,
        stage_name=stage_name,
        config_name=name,
    )
    _assert_control_stage(stage, name=f"{name}.training stage")
    return llm_path, control_cfg


def _exact_mixture_resume_contract(
    cfg: Mapping[str, Any],
    *,
    stage_idx: int,
    stage_name: str,
    name: str,
) -> dict[str, Any] | None:
    """Return the sample/step-order contract for control mixture training.

    Model/optimizer tensors alone are insufficient for a mid-epoch resume.  A
    changed root order, mixture plan, effective batch, schedule, or EMA policy
    can otherwise pair restored optimizer state with a different next sample or
    learning-rate trajectory.
    """

    raw_data = cfg.get("data") or {}
    if not isinstance(raw_data, Mapping):
        raise HeatmapControlResumeError(f"{name}.data must be a mapping")
    if raw_data.get("dataset_type") != "expert_dagger_mixture":
        return None

    runtime = _require_mapping(cfg.get("runtime"), name=f"{name}.runtime")
    model_cfg = _require_mapping(cfg.get("model"), name=f"{name}.model")
    llm_cfg = _require_mapping(
        model_cfg.get("llm"), name=f"{name}.model.llm"
    )
    try:
        native_dependency = validate_native_internnav_dependency_contract(
            runtime.get(NATIVE_DEPENDENCY_RUNTIME_KEY),
            expected_model_path=llm_cfg.get("model_path"),
            name=f"{name}.runtime.{NATIVE_DEPENDENCY_RUNTIME_KEY}",
        )
    except NativeInternNavDependencyError as exc:
        raise HeatmapControlResumeError(str(exc)) from exc

    mixture = _require_mapping(
        raw_data.get("mixture"), name=f"{name}.data.mixture"
    )
    dagger = _require_mapping(
        raw_data.get("trajectory_dagger"),
        name=f"{name}.data.trajectory_dagger",
    )
    trajectory = _require_mapping(
        raw_data.get("trajectory"), name=f"{name}.data.trajectory"
    )
    roots = dagger.get("collection_roots")
    if (
        not isinstance(roots, (list, tuple))
        or not roots
        or any(not isinstance(root, str) or not root for root in roots)
    ):
        raise HeatmapControlResumeError(
            f"{name}.data.trajectory_dagger.collection_roots must be a "
            "non-empty ordered list"
        )
    if len(set(roots)) != len(roots):
        raise HeatmapControlResumeError(
            f"{name}.data.trajectory_dagger.collection_roots contains duplicates"
        )
    if raw_data.get("in_order") is not True:
        raise HeatmapControlResumeError(
            f"{name}.data.in_order must be true for exact mixture resume"
        )

    epoch_size = mixture.get("epoch_size")
    seed = mixture.get("seed")
    if isinstance(epoch_size, bool) or not isinstance(epoch_size, int) or epoch_size < 1:
        raise HeatmapControlResumeError(
            f"{name}.data.mixture.epoch_size must be a positive integer"
        )
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise HeatmapControlResumeError(
            f"{name}.data.mixture.seed must be an integer"
        )

    optim = _require_mapping(cfg.get("optim"), name=f"{name}.optim")
    batch_size = optim.get("batch_size")
    grad_accum_steps = optim.get("grad_accum_steps")
    if (
        isinstance(batch_size, bool)
        or not isinstance(batch_size, int)
        or batch_size < 1
        or isinstance(grad_accum_steps, bool)
        or not isinstance(grad_accum_steps, int)
        or grad_accum_steps < 1
    ):
        raise HeatmapControlResumeError(
            f"{name}.optim batch_size/grad_accum_steps must be positive integers"
        )

    log_cfg = _require_mapping(cfg.get("log"), name=f"{name}.log")
    mid_epoch_save_every = log_cfg.get("mid_epoch_save_every", 0)
    if (
        isinstance(mid_epoch_save_every, bool)
        or not isinstance(mid_epoch_save_every, int)
        or mid_epoch_save_every < 0
    ):
        raise HeatmapControlResumeError(
            f"{name}.log.mid_epoch_save_every must be a non-negative integer"
        )
    if (
        mid_epoch_save_every > 0
        and mid_epoch_save_every % grad_accum_steps != 0
    ):
        raise HeatmapControlResumeError(
            f"{name}.log.mid_epoch_save_every must align with grad_accum_steps"
        )

    stage = _resolve_checkpoint_stage(
        cfg,
        stage_idx=stage_idx,
        stage_name=stage_name,
        config_name=name,
    )
    stage_epochs = stage.get("epochs")
    if isinstance(stage_epochs, bool) or not isinstance(stage_epochs, int) or stage_epochs < 1:
        raise HeatmapControlResumeError(
            f"{name}.training stage epochs must be a positive integer"
        )

    gpu_cfg = cfg.get("gpu") or {}
    if not isinstance(gpu_cfg, Mapping):
        raise HeatmapControlResumeError(f"{name}.gpu must be a mapping")
    devices = gpu_cfg.get("devices")
    configured_world_size = (
        len(devices) if isinstance(devices, (list, tuple)) and devices else None
    )

    optimizer_keys = (
        "optimizer",
        "learning_rate",
        "heatmap_tokenizer_lr",
        "heatmap_control_lr",
        "heatmap_gate_lr",
        "weight_decay",
        "grad_clip",
        "amp",
        "scheduler",
        "warmup_ratio",
        "min_lr",
        "batch_size",
        "grad_accum_steps",
        "ema_decay",
        "ema_warmup_steps",
    )
    return {
        "schema": "heatmap-control-exact-mixture-resume-v1",
        "global_seed": cfg.get("seed"),
        "native_internnav_dependency": native_dependency,
        "data": {
            "root": raw_data.get("root"),
            "train_split": raw_data.get("train_split"),
            "image_size": raw_data.get("image_size"),
            "init_hm_size": raw_data.get("init_hm_size"),
            "in_order": True,
            "trajectory": dict(trajectory),
            "trajectory_dagger": dict(dagger),
            "collection_roots": list(roots),
            "expected_policy_fingerprint": dagger.get(
                "expected_policy_fingerprint"
            ),
            "mixture": dict(mixture),
        },
        "optim": {key: optim.get(key) for key in optimizer_keys},
        "stage_epochs": stage_epochs,
        "configured_world_size": configured_world_size,
        "mid_epoch_save_every": mid_epoch_save_every,
    }


def _validate_mixture_sampler_state(
    checkpoint: Mapping[str, Any],
    contract: Mapping[str, Any],
) -> None:
    state = _require_mapping(
        checkpoint.get("mixture_sampler_state"),
        name="mixture_sampler_state",
    )
    mixture = contract["data"]["mixture"]
    expected_profile = mixture.get("profile")
    expected_weights = mixture.get("weights")
    if expected_weights is None:
        if expected_profile != "expert50_normal20_hard30":
            raise HeatmapControlResumeError(
                "unsupported exact-resume mixture profile: "
                f"{expected_profile!r}"
            )
        expected_weights = {
            "expert": 0.5,
            "dagger_normal": 0.2,
            "dagger_hard": 0.3,
        }
    else:
        total = float(sum(expected_weights.values()))
        expected_weights = {
            key: float(value) / total
            for key, value in expected_weights.items()
        }
        expected_profile = "custom"

    expected = {
        "schema": "heatmapvln-deterministic-mixture-sampler-v1",
        "seed": mixture["seed"],
        "requested_epoch_size": mixture["epoch_size"],
        "global_epoch_size": mixture["epoch_size"],
        "drop_last": True,
        "profile": expected_profile,
        "weights": expected_weights,
    }
    mismatches = {
        key: {"expected": value, "actual": state.get(key)}
        for key, value in expected.items()
        if state.get(key) != value
    }
    checkpoint_epoch = checkpoint.get("epoch")
    if state.get("epoch") != checkpoint_epoch:
        mismatches["epoch"] = {
            "expected": checkpoint_epoch,
            "actual": state.get("epoch"),
        }
    configured_world_size = contract.get("configured_world_size")
    if (
        configured_world_size is not None
        and state.get("num_replicas") != configured_world_size
    ):
        mismatches["num_replicas"] = {
            "expected": configured_world_size,
            "actual": state.get("num_replicas"),
        }
    if state.get("rank") != 0:
        mismatches["rank"] = {"expected": 0, "actual": state.get("rank")}

    batch = checkpoint.get("batch")
    if batch is not None:
        replicas = state.get("num_replicas")
        batch_size = contract["optim"]["batch_size"]
        if (
            isinstance(batch, bool)
            or not isinstance(batch, int)
            or batch < 1
            or not isinstance(replicas, int)
            or replicas < 1
            or batch >= mixture["epoch_size"] // replicas // batch_size
        ):
            mismatches["batch"] = {
                "expected": (
                    "an absolute consumed batch strictly before the epoch end"
                ),
                "actual": batch,
            }
        grad_accum_steps = contract["optim"]["grad_accum_steps"]
        if isinstance(batch, int) and batch % grad_accum_steps != 0:
            mismatches["batch_accumulation_boundary"] = {
                "expected_multiple": grad_accum_steps,
                "actual": batch,
            }
    if mismatches:
        raise HeatmapControlResumeError(
            "mixture sampler state is incompatible with exact resume: "
            f"{mismatches}"
        )


def _load_weights_only(path: Path, *, purpose: str) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise HeatmapControlResumeError(
            f"{purpose} requires torch.load(..., weights_only=True)"
        ) from exc
    except Exception as exc:
        raise HeatmapControlResumeError(
            f"unable to safely deserialize {purpose}: {path}"
        ) from exc


def _inspect_current_dependency(
    cfg: Mapping[str, Any],
    control_cfg: Mapping[str, Any],
) -> dict[str, Any]:
    configured_path = control_cfg.get("heatmap_checkpoint_path")
    configured_sha = control_cfg.get("heatmap_checkpoint_sha256")
    if not isinstance(configured_sha, str) or _SHA256_PATTERN.fullmatch(
        configured_sha
    ) is None:
        raise HeatmapControlResumeError(
            "current heatmap_checkpoint_sha256 must be 64 lowercase hex characters"
        )
    try:
        path = Path(configured_path).expanduser().resolve(strict=True)
        actual_sha = compute_file_sha256(path)
    except Exception as exc:
        raise HeatmapControlResumeError(
            f"unable to inspect current frozen heatmap dependency: {configured_path}"
        ) from exc
    if actual_sha != configured_sha:
        raise HeatmapControlResumeError(
            "current frozen heatmap dependency SHA-256 mismatch: "
            f"configured={configured_sha}, actual={actual_sha}"
        )

    payload = _load_weights_only(path, purpose="frozen heatmap dependency")
    if not isinstance(payload, Mapping):
        raise HeatmapControlResumeError(
            "frozen heatmap dependency payload must be a mapping"
        )
    raw_state = payload.get(FROZEN_HEATMAP_STATE_KEY)
    if not isinstance(raw_state, Mapping) or not raw_state:
        raise HeatmapControlResumeError(
            "frozen heatmap dependency must contain a non-empty "
            f"{FROZEN_HEATMAP_STATE_KEY}"
        )
    canonical_names: set[str] = set()
    for raw_name, value in raw_state.items():
        name = _canonical_key(raw_name)
        if name in canonical_names:
            raise HeatmapControlResumeError(
                "frozen heatmap dependency has duplicate canonical key: " + name
            )
        if not name.startswith(FROZEN_HEATMAP_PREFIX) or not torch.is_tensor(value):
            raise HeatmapControlResumeError(
                "frozen heatmap dependency contains a non-heatmap/non-tensor entry: "
                + name
            )
        canonical_names.add(name)

    contract = {
        "schema_version": DEPENDENCY_SCHEMA,
        "checkpoint_sha256": actual_sha,
        "target_module": "heatmap_vln",
        "frozen": True,
        "tensor_count": len(canonical_names),
    }
    runtime = cfg.get("runtime") or {}
    if isinstance(runtime, Mapping) and runtime.get("frozen_heatmap_dependency") is not None:
        _assert_dependency_matches(
            runtime.get("frozen_heatmap_dependency"),
            contract,
            name="current config runtime frozen_heatmap_dependency",
        )
    return contract


def _assert_dependency_matches(
    candidate: Any,
    expected: Mapping[str, Any],
    *,
    name: str,
) -> None:
    dependency = _require_mapping(candidate, name=name)
    required_fields = (
        "schema_version",
        "checkpoint_sha256",
        "target_module",
        "frozen",
        "tensor_count",
    )
    mismatches = {
        field: {"expected": expected[field], "actual": dependency.get(field)}
        for field in required_fields
        if dependency.get(field) != expected[field]
    }
    tensor_count = dependency.get("tensor_count")
    if isinstance(tensor_count, bool) or not isinstance(tensor_count, int) or tensor_count < 1:
        mismatches["tensor_count"] = {
            "expected": expected["tensor_count"],
            "actual": tensor_count,
        }
    if mismatches:
        raise HeatmapControlResumeError(
            f"{name} does not match the current frozen heatmap dependency: "
            f"{mismatches}"
        )


def _validate_payload(
    payload: Any,
    model: nn.Module,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    checkpoint = _require_mapping(payload, name="resume checkpoint")
    try:
        stage_idx = int(checkpoint.get("stage_idx", 0))
    except (TypeError, ValueError) as exc:
        raise HeatmapControlResumeError(
            "resume checkpoint stage_idx must be an integer"
        ) from exc
    stage_name_value = checkpoint.get("stage_name", "")
    if not isinstance(stage_name_value, str):
        raise HeatmapControlResumeError("resume checkpoint stage_name must be a string")

    current_path, current_control = _assert_native_config(
        cfg,
        name="current config",
        stage_idx=stage_idx,
        stage_name=stage_name_value,
    )
    saved_cfg = _require_mapping(checkpoint.get("config"), name="saved config")
    saved_path, saved_control = _assert_native_config(
        saved_cfg,
        name="saved config",
        stage_idx=stage_idx,
        stage_name=stage_name_value,
    )
    if saved_path != current_path:
        raise HeatmapControlResumeError(
            "saved and current configs do not use the same original InternNav path: "
            f"saved={saved_path!r}, current={current_path!r}"
        )
    architecture_mismatches = {
        field: {
            "saved": saved_control.get(field, default),
            "current": current_control.get(field, default),
        }
        for field, default in _CONTROL_ARCH_DEFAULTS.items()
        if saved_control.get(field, default) != current_control.get(field, default)
    }
    if architecture_mismatches:
        raise HeatmapControlResumeError(
            "saved heatmap-control architecture differs from the current config: "
            f"{architecture_mismatches}"
        )

    current_training_contract = _exact_mixture_resume_contract(
        cfg,
        stage_idx=stage_idx,
        stage_name=stage_name_value,
        name="current config",
    )
    saved_training_contract = _exact_mixture_resume_contract(
        saved_cfg,
        stage_idx=stage_idx,
        stage_name=stage_name_value,
        name="saved config",
    )
    if saved_training_contract != current_training_contract:
        raise HeatmapControlResumeError(
            "saved and current exact-resume data/optimizer contracts differ: "
            f"saved={saved_training_contract!r}, "
            f"current={current_training_contract!r}"
        )
    if current_training_contract is not None:
        _validate_mixture_sampler_state(checkpoint, current_training_contract)

    current_dependency = _inspect_current_dependency(cfg, current_control)
    saved_runtime = _require_mapping(
        saved_cfg.get("runtime"), name="saved config.runtime"
    )
    _assert_dependency_matches(
        saved_runtime.get("frozen_heatmap_dependency"),
        current_dependency,
        name="saved frozen_heatmap_dependency",
    )
    if saved_control.get("heatmap_checkpoint_sha256") != current_dependency[
        "checkpoint_sha256"
    ]:
        raise HeatmapControlResumeError(
            "saved heatmap-control checkpoint SHA does not match the current dependency"
        )

    requires_scaler = _validate_training_runtime_states(checkpoint, cfg, saved_cfg)
    expected = _expected_control_parameters(model)
    _validate_tensor_state(
        checkpoint.get("trainable_state_dict"),
        expected,
        state_name="trainable_state_dict",
    )
    _validate_tensor_state(
        checkpoint.get("online_trainable_state_dict"),
        expected,
        state_name="online_trainable_state_dict",
    )

    ema_state = _require_mapping(
        checkpoint.get("ema_state_dict"), name="ema_state_dict"
    )
    _validate_tensor_state(
        ema_state.get("shadow"),
        expected,
        state_name="ema_state_dict.shadow",
    )

    return {
        "schema_version": "heatmap-control-resume-validation-v1",
        "state_tensor_count": len(expected),
        "validated_trainable_state": True,
        "validated_online_state": True,
        "validated_ema_shadow": True,
        "validated_optimizer_state": True,
        "validated_scheduler_state": True,
        "validated_scaler_state": requires_scaler,
        "validated_exact_training_contract": (
            current_training_contract is not None
        ),
        "validated_mixture_sampler_state": (
            current_training_contract is not None
        ),
        "native_model_path": current_path,
        "frozen_heatmap_dependency": dict(current_dependency),
    }


def validate_heatmap_control_resume_checkpoint(
    checkpoint_path: str | os.PathLike[str],
    model: nn.Module,
    cfg: Mapping[str, Any],
) -> dict[str, Any]:
    """Validate a control-only resume checkpoint without mutating ``model``."""
    if not heatmap_control_enabled(cfg):
        raise HeatmapControlResumeError(
            "heatmap-control resume validation requires heatmap_control.enabled=true"
        )
    try:
        path = Path(checkpoint_path).expanduser().resolve(strict=True)
    except Exception as exc:
        raise HeatmapControlResumeError(
            f"resume checkpoint does not exist: {checkpoint_path}"
        ) from exc
    payload = _load_weights_only(path, purpose="heatmap-control resume checkpoint")
    report = _validate_payload(payload, model, cfg)
    report["checkpoint_path"] = str(path)
    return report


__all__ = [
    "HeatmapControlResumeError",
    "heatmap_control_enabled",
    "reject_heatmap_control_load_weights",
    "validate_heatmap_control_resume_checkpoint",
]

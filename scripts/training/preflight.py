"""Real one-batch training preflight used by ``scripts/train.py``."""

from __future__ import annotations

import math
from typing import Any

import torch

from .train_loop import train_one_epoch
from .pose_adaptation_smoke import (
    EXPECTED_BATCH_PER_RANK,
    build_local_rank_audit,
    expected_smoke_world_size,
    gather_and_validate_local_audit,
    install_gradient_hooks,
    smoke_audit_enabled,
)
from .past_plan_action_smoke import (
    EXPECTED_BATCH_PER_RANK as PPA_EXPECTED_BATCH_PER_RANK,
    build_local_rank_audit as build_ppa_local_rank_audit,
    expected_smoke_world_size as expected_ppa_smoke_world_size,
    gather_and_validate_local_audit as gather_and_validate_ppa_local_audit,
    install_gradient_hooks as install_ppa_gradient_hooks,
    smoke_audit_enabled as ppa_smoke_audit_enabled,
)


def assert_single_view_training_contract(
    model,
    optimizer: torch.optim.Optimizer,
    cfg: dict,
    stage_cfg: dict,
) -> dict[str, int] | None:
    """Fail closed for the frozen-native single-view heatmap architecture."""

    heatmap_cfg = cfg.get("model", {}).get("heatmap", {})
    if str(heatmap_cfg.get("input_mode", "panoramic")) != "internnav_single_view":
        return None

    llm_cfg = cfg.get("model", {}).get("llm", {})
    if llm_cfg.get("use_lora") is not False:
        raise RuntimeError("internnav_single_view requires model.llm.use_lora=false")
    if heatmap_cfg.get("heatmap_trains_backbone") is not False:
        raise RuntimeError(
            "internnav_single_view requires "
            "model.heatmap.heatmap_trains_backbone=false"
        )
    if heatmap_cfg.get("feature_source") != "vit_only":
        raise RuntimeError(
            "internnav_single_view requires model.heatmap.feature_source=vit_only"
        )
    if heatmap_cfg.get("architecture_id") != (
        "internnav_single_view_vision_only_four_direction_v2"
    ):
        raise RuntimeError("single-view heatmap architecture_id mismatch")
    if tuple(heatmap_cfg.get("output_direction_order", ())) != (
        "front",
        "right",
        "back",
        "left",
    ):
        raise RuntimeError(
            "single-view output direction order must be "
            "front/right/back/left"
        )
    if heatmap_cfg.get("history_pose_convention") != (
        "habitat_c2w_minus_z__forward_left_cos_yaw_sin_yaw__v1"
    ):
        raise RuntimeError("single-view history pose convention is not explicit/correct")
    trajectory_cfg = heatmap_cfg.get("trajectory") or {}
    expected_trajectory = {
        "enable": True,
        "num_freqs": 16,
        "d_attn": 256,
        "num_heads": 4,
        "num_layers": 2,
        "max_spatial_range": 10.0,
    }
    mismatched_trajectory = {
        key: {"expected": expected, "actual": trajectory_cfg.get(key)}
        for key, expected in expected_trajectory.items()
        if trajectory_cfg.get(key) != expected
    }
    if mismatched_trajectory:
        raise RuntimeError(
            "single-view pose-projection migration requires the audited "
            f"trajectory encoding: {mismatched_trajectory}"
        )
    control_cfg = (
        cfg.get("model", {})
        .get("action_head", {})
        .get("nextdit", {})
        .get("heatmap_control", {})
    )
    if control_cfg.get("enabled", False):
        expected_trainable = ["heatmap_tokenizer", "heatmap_control"]
        if list(stage_cfg.get("trainable_modules", ())) != expected_trainable:
            raise RuntimeError(
                f"heatmap control must train exactly {expected_trainable}"
            )
        if stage_cfg.get("strict_trainable_modules") is not True:
            raise RuntimeError("heatmap control requires strict_trainable_modules=true")
        if not stage_cfg.get("train_action", False):
            raise RuntimeError("heatmap control requires train_action=true")
        if stage_cfg.get("train_history", False) or stage_cfg.get("train_future", False):
            raise RuntimeError("heatmap control must not train the frozen heatmap head")
        if bool(stage_cfg.get("train_lm", stage_cfg.get("train_system2_sft", False))):
            raise RuntimeError("heatmap control must keep native System2 frozen")

        module = _unwrap_model(model)
        if getattr(module, "pano_latent_adapter", None) is not None:
            raise RuntimeError("heatmap control forbids panoramic adapters")
        heatmap_head = getattr(module, "heatmap_vln", None)
        if heatmap_head is None:
            raise RuntimeError("heatmap control requires a loaded frozen heatmap head")
        if any(parameter.requires_grad for parameter in heatmap_head.parameters()):
            raise RuntimeError("the pretrained heatmap dependency must remain frozen")
        dependency = cfg.get("runtime", {}).get("frozen_heatmap_dependency")
        expected_sha256 = control_cfg.get("heatmap_checkpoint_sha256")
        if (
            not isinstance(dependency, dict)
            or dependency.get("schema_version")
            != "frozen-heatmap-checkpoint-v1"
            or dependency.get("checkpoint_sha256") != expected_sha256
            or dependency.get("target_module") != "heatmap_vln"
            or dependency.get("frozen") is not True
        ):
            raise RuntimeError(
                "frozen heatmap dependency metadata is missing or does not "
                "match heatmap_control.heatmap_checkpoint_sha256"
            )
        if dependency.get("tensor_count") != len(
            dict(heatmap_head.named_parameters())
        ):
            raise RuntimeError(
                "frozen heatmap dependency tensor count no longer matches "
                "the constructed heatmap head"
            )
        tokenizer = getattr(module, "heatmap_tokenizer", None)
        action_head = getattr(module, "nextdit_action_head", None)
        adapters_fn = getattr(action_head, "heatmap_control_adapters", None)
        adapters = tuple(adapters_fn()) if callable(adapters_fn) else ()
        expected_layers = int(
            cfg["model"]["action_head"]["nextdit"].get("dit_layers", 12)
        )
        if tokenizer is None or len(adapters) != expected_layers:
            raise RuntimeError(
                "heatmap control construction is incomplete: "
                f"tokenizer={tokenizer is not None}, adapters={len(adapters)}"
            )
        allowed_names = {
            name
            for name, parameter in module.named_parameters()
            if (
                name.startswith("heatmap_tokenizer.")
                or (
                    name.startswith("nextdit_action_head.traj_dit.model.layers.")
                    and ".heatmap_control." in name
                )
            )
            and parameter.requires_grad
        }
        all_trainable = {
            name
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
        if not allowed_names or all_trainable != allowed_names:
            raise RuntimeError(
                "only tokenizer and per-layer control may be trainable: "
                f"unexpected={sorted(all_trainable - allowed_names)[:8]}"
            )
        optimizer_ids = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        trainable_ids = {
            id(parameter)
            for parameter in module.parameters()
            if parameter.requires_grad
        }
        if len(optimizer_ids) != len(set(optimizer_ids)):
            raise RuntimeError("heatmap control optimizer contains duplicate parameters")
        if set(optimizer_ids) != trainable_ids:
            raise RuntimeError("heatmap control optimizer coverage mismatch")
        invalid_groups = [
            group.get("name")
            for group in optimizer.param_groups
            if not str(group.get("name", "")).startswith(
                ("heatmap_tokenizer", "heatmap_control")
            )
        ]
        if invalid_groups:
            raise RuntimeError(
                f"heatmap control optimizer contains native groups: {invalid_groups}"
            )
        return {
            "trainable_parameter_tensors": len(trainable_ids),
            "optimizer_parameter_tensors": len(optimizer_ids),
            "optimizer_groups": len(optimizer.param_groups),
            "control_adapters": len(adapters),
        }

    ppa_cfg = cfg.get("model", {}).get("past_plan_action") or {}
    if ppa_cfg.get("enabled", False):
        expected_trainable = ["past_plan_action", "heatmap_vln"]
        if list(stage_cfg.get("trainable_modules", ())) != expected_trainable:
            raise RuntimeError(
                f"Past->Plan->Action must train exactly {expected_trainable}"
            )
        if stage_cfg.get("strict_trainable_modules") is not True:
            raise RuntimeError("Past->Plan->Action requires strict trainable scope")
        if not stage_cfg.get("train_future", False):
            raise RuntimeError("Past->Plan->Action requires Future supervision")
        if bool(stage_cfg.get("train_lm", stage_cfg.get("train_system2_sft", False))):
            raise RuntimeError("Past->Plan->Action keeps native System2 frozen")

        module = _unwrap_model(model)
        chain = getattr(module, "past_plan_action", None)
        head = getattr(module, "heatmap_vln", None)
        action_head = getattr(module, "nextdit_action_head", None)
        if chain is None or head is None or action_head is None:
            raise RuntimeError("Past->Plan->Action construction is incomplete")
        if getattr(module, "pano_latent_adapter", None) is not None:
            raise RuntimeError("Past->Plan->Action forbids panoramic adapters")
        if getattr(module, "_heatmap_control_enabled", False):
            raise RuntimeError("Past->Plan->Action forbids legacy heatmap control")
        frozen_modules = (
            getattr(module, "qwen2_5_vl", None),
            action_head,
            action_head.cond_projector,
        )
        if any(
            submodule is None
            or submodule.training
            or any(parameter.requires_grad for parameter in submodule.parameters())
            for submodule in frozen_modules
        ):
            raise RuntimeError(
                "Past->Plan->Action native Qwen/Plan projector/NextDiT must be frozen eval"
            )
        if module.latent_queries.requires_grad:
            raise RuntimeError("Past->Plan->Action native TRAJ queries must be frozen")

        allowed_past_prefixes = (
            "heatmap_vln.coarse.proj_history.",
            "heatmap_vln.coarse.proj_traj.",
            "heatmap_vln.coarse.pos_embed",
            "heatmap_vln.coarse.self_attn.",
            "heatmap_vln.coarse.heatmap_head.",
            "heatmap_vln.coarse.vis_head.",
            "heatmap_vln.fine.",
        )
        trainable_names = {
            name
            for name, parameter in module.named_parameters()
            if parameter.requires_grad
        }
        invalid_names = sorted(
            name
            for name in trainable_names
            if not name.startswith("past_plan_action.")
            and not any(
                name == prefix or name.startswith(prefix)
                for prefix in allowed_past_prefixes
            )
        )
        if invalid_names or not trainable_names:
            raise RuntimeError(
                "Past->Plan->Action trainable scope mismatch: "
                f"invalid={invalid_names[:8]}"
            )
        optimizer_ids = [
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group["params"]
        ]
        trainable_ids = {
            id(parameter)
            for parameter in module.parameters()
            if parameter.requires_grad
        }
        if len(optimizer_ids) != len(set(optimizer_ids)):
            raise RuntimeError("Past->Plan->Action optimizer has duplicate tensors")
        if set(optimizer_ids) != trainable_ids:
            raise RuntimeError("Past->Plan->Action optimizer coverage mismatch")
        invalid_groups = [
            group.get("name")
            for group in optimizer.param_groups
            if not str(group.get("name", "")).startswith(
                ("past_plan_action_", "heatmap_")
            )
        ]
        if invalid_groups:
            raise RuntimeError(
                f"Past->Plan->Action optimizer contains native groups: {invalid_groups}"
            )
        return {
            "trainable_parameter_tensors": len(trainable_ids),
            "optimizer_parameter_tensors": len(optimizer_ids),
            "optimizer_groups": len(optimizer.param_groups),
        }

    warmstart = stage_cfg.get("heatmap_warmstart_contract") or {}
    if warmstart.get("policy") != "internnav_single_view_head_v2":
        raise RuntimeError("single-view warm-start policy mismatch")
    if list(stage_cfg.get("trainable_modules", ())) != ["heatmap_vln"]:
        raise RuntimeError(
            "internnav_single_view must train exactly ['heatmap_vln']"
        )
    if stage_cfg.get("strict_trainable_modules") is not True:
        raise RuntimeError("internnav_single_view requires strict_trainable_modules=true")
    if stage_cfg.get("train_action", False) or bool(
        stage_cfg.get("train_lm", stage_cfg.get("train_system2_sft", False))
    ):
        raise RuntimeError("internnav_single_view stage must not train System1/System2")

    module = _unwrap_model(model)
    if getattr(module, "pano_latent_adapter", None) is not None:
        raise RuntimeError("internnav_single_view model contains a panoramic adapter")
    if getattr(module, "nextdit_action_head", None) is not None:
        raise RuntimeError(
            "heatmap-only training must not instantiate System1; deployment "
            "loads untouched native System1 separately"
        )
    suspicious = sorted(
        {
            name
            for name, _parameter in module.named_parameters()
            if "lora_" in name.lower()
        }
        | {
            name
            for name, child in module.named_modules()
            if "lora" in name.lower() or "lora" in type(child).__name__.lower()
        }
    )
    if suspicious:
        raise RuntimeError(
            "internnav_single_view model contains forbidden LoRA/PEFT state: "
            f"{suspicious[:8]}"
        )
    head = getattr(module, "heatmap_vln", None)
    if head is None or getattr(head, "llm_dpt_fusion", None) is not None:
        raise RuntimeError("single-view model must own a head with no LLM-DPT path")
    if getattr(head, "architecture_id", None) != heatmap_cfg["architecture_id"]:
        raise RuntimeError("constructed single-view head architecture_id mismatch")
    if tuple(getattr(head, "output_direction_order", ())) != tuple(
        heatmap_cfg["output_direction_order"]
    ):
        raise RuntimeError("constructed single-view head direction order mismatch")
    if getattr(head, "history_pose_convention", None) != heatmap_cfg[
        "history_pose_convention"
    ]:
        raise RuntimeError("constructed single-view head pose convention mismatch")

    named = dict(module.named_parameters())
    trainable_ids = {
        id(parameter)
        for name, parameter in named.items()
        if parameter.requires_grad and name.startswith("heatmap_vln.")
    }
    all_trainable_ids = {
        id(parameter) for parameter in named.values() if parameter.requires_grad
    }
    if not trainable_ids or trainable_ids != all_trainable_ids:
        violations = [
            name
            for name, parameter in named.items()
            if parameter.requires_grad and not name.startswith("heatmap_vln.")
        ]
        raise RuntimeError(
            "only heatmap_vln may be trainable; violations="
            f"{violations[:8]}"
        )
    optimizer_ids: list[int] = []
    for index, group in enumerate(optimizer.param_groups):
        group_name = group.get("name")
        if not isinstance(group_name, str) or not group_name.startswith("heatmap_"):
            raise RuntimeError(
                f"optimizer group {index} is not heatmap-only: {group_name!r}"
            )
        optimizer_ids.extend(id(parameter) for parameter in group["params"])
    if len(optimizer_ids) != len(set(optimizer_ids)):
        raise RuntimeError("single-view optimizer contains duplicate parameters")
    if set(optimizer_ids) != trainable_ids:
        raise RuntimeError(
            "single-view optimizer does not exactly cover trainable heatmap "
            f"parameters: missing={len(trainable_ids - set(optimizer_ids))}, "
            f"extra={len(set(optimizer_ids) - trainable_ids)}"
        )
    return {
        "trainable_parameter_tensors": len(trainable_ids),
        "optimizer_parameter_tensors": len(optimizer_ids),
        "optimizer_groups": len(optimizer.param_groups),
    }


def _unwrap_model(model):
    """Return the underlying module without depending on a DDP import."""
    return getattr(model, "module", model)


def _snapshot_trainable_heatmap_parameters(
    model,
    stage_cfg: dict,
) -> dict[str, torch.Tensor]:
    """Copy the small trainable heatmap/control surface for a delta check."""
    trainable = set(stage_cfg.get("trainable_modules", ()))
    tracks_head = "heatmap_vln" in trainable
    tracks_control = bool(
        {"heatmap_tokenizer", "heatmap_control"} & trainable
    )
    if not tracks_head and not tracks_control:
        return {}

    module = _unwrap_model(model)
    if not hasattr(module, "named_parameters"):
        raise RuntimeError(
            "Heatmap training preflight cannot inspect model parameters"
        )

    def selected(name: str) -> bool:
        if tracks_head and name.startswith("heatmap_vln."):
            return True
        return tracks_control and (
            name.startswith("heatmap_tokenizer.")
            or (
                name.startswith("nextdit_action_head.traj_dit.model.layers.")
                and ".heatmap_control." in name
            )
        )

    snapshot = {
        name: param.detach().to(device="cpu", copy=True)
        for name, param in module.named_parameters()
        if selected(name) and param.requires_grad
    }
    if not snapshot:
        raise RuntimeError(
            "Heatmap training preflight found no selected trainable parameters"
        )
    return snapshot


def _snapshot_heatmap_control_gates(
    model,
    stage_cfg: dict,
    cfg: dict,
) -> dict[str, torch.Tensor]:
    """Snapshot every zero-initialized per-layer control gate.

    A generic parameter-delta check is insufficient here: AdamW weight decay
    can move tokenizer/attention weights even when the trajectory objective is
    disconnected. The gates have zero weight decay and start at exactly zero,
    so a non-zero gate delta after one step proves that the flow-matching loss
    reached at least one control branch.
    """
    if "heatmap_control" not in set(stage_cfg.get("trainable_modules", ())):
        return {}

    module = _unwrap_model(model)
    gates = {
        name: parameter.detach().to(device="cpu", copy=True)
        for name, parameter in module.named_parameters()
        if (
            name.startswith("nextdit_action_head.traj_dit.model.layers.")
            and ".heatmap_control." in name
            and name.endswith(".gate")
            and parameter.requires_grad
        )
    }
    expected_layers = int(
        cfg.get("model", {})
        .get("action_head", {})
        .get("nextdit", {})
        .get("dit_layers", 0)
        or 0
    )
    if not gates or (expected_layers and len(gates) != expected_layers):
        raise RuntimeError(
            "Heatmap-control preflight expected one trainable zero gate per "
            f"NextDiT layer: expected={expected_layers}, found={len(gates)}"
        )
    invalid = [
        name
        for name, value in gates.items()
        if not bool(torch.isfinite(value.float()).all())
        or bool(torch.count_nonzero(value).item())
    ]
    if invalid:
        raise RuntimeError(
            "Heatmap-control dry-run must start from finite, exactly zero gates; "
            f"invalid={invalid[:5]}"
        )
    return gates


def _verify_heatmap_control_gate_delta(
    model,
    before: dict[str, torch.Tensor],
) -> dict[str, float]:
    """Require a real optimizer update on at least one zero-init gate."""
    if not before:
        return {}

    parameters = dict(_unwrap_model(model).named_parameters())
    missing = sorted(set(before) - set(parameters))
    if missing:
        raise RuntimeError(
            "Heatmap-control preflight lost snapshotted gates: "
            f"{missing[:5]}"
        )

    changed_gates = 0
    changed_elements = 0
    max_abs_delta = 0.0
    for name, original in before.items():
        current = parameters[name].detach().to(device="cpu", copy=True)
        if current.shape != original.shape or current.dtype != original.dtype:
            raise RuntimeError(
                "Heatmap-control gate metadata changed during preflight: "
                f"{name}"
            )
        if not bool(torch.isfinite(current.float()).all()):
            raise RuntimeError(
                "Heatmap-control preflight produced a non-finite gate: " + name
            )
        delta = current.float() - original.float()
        count = int(torch.count_nonzero(delta).item())
        if count:
            changed_gates += 1
            changed_elements += count
            max_abs_delta = max(max_abs_delta, float(delta.abs().max().item()))

    if changed_gates == 0 or changed_elements == 0 or max_abs_delta <= 0.0:
        raise RuntimeError(
            "Heatmap-control preflight did not update any zero-initialized gate; "
            "the GT trajectory flow-matching loss is disconnected, invalid, or "
            "the batch contains no usable control supervision"
        )
    return {
        "heatmap_control_changed_gates": float(changed_gates),
        "heatmap_control_changed_gate_elements": float(changed_elements),
        "heatmap_control_gate_delta_max_abs": max_abs_delta,
    }


def _evenly_spaced_indices(numel: int, sample_count: int) -> list[int]:
    if numel <= sample_count:
        return list(range(numel))
    if sample_count == 1:
        return [0]
    return [
        round(index * (numel - 1) / (sample_count - 1))
        for index in range(sample_count)
    ]


def _snapshot_frozen_lora_samples(
    model,
    stage_cfg: dict,
    *,
    samples_per_tensor: int = 8,
) -> dict[str, tuple[list[int], torch.Tensor]]:
    """Take a tiny deterministic sample from each frozen LoRA tensor.

    This is an integrity sentinel, not a replacement for the strict trainable
    scope check.  Sampling every frozen LoRA tensor avoids a full backbone
    clone while still catching accidental optimizer/forward mutations across
    layers and projections.
    """
    trainable_modules = set(stage_cfg.get("trainable_modules", ()))
    if (
        "heatmap_vln" not in trainable_modules
        or trainable_modules & {"lora", "vlm_lora"}
    ):
        return {}

    module = _unwrap_model(model)
    if not hasattr(module, "named_parameters"):
        return {}

    snapshot: dict[str, tuple[list[int], torch.Tensor]] = {}
    for name, param in module.named_parameters():
        if "lora_" not in name.lower() or param.requires_grad or param.numel() == 0:
            continue
        indices = _evenly_spaced_indices(
            param.numel(),
            min(samples_per_tensor, param.numel()),
        )
        device_indices = torch.tensor(
            indices,
            dtype=torch.long,
            device=param.device,
        )
        values = (
            param.detach()
            .reshape(-1)
            .index_select(0, device_indices)
            .to(device="cpu", copy=True)
        )
        snapshot[name] = (indices, values)
    return snapshot


def _verify_trainable_heatmap_delta(
    model,
    before: dict[str, torch.Tensor],
) -> dict[str, float]:
    if not before:
        return {}

    module = _unwrap_model(model)
    parameters = dict(module.named_parameters())
    missing = sorted(set(before) - set(parameters))
    if missing:
        raise RuntimeError(
            "Heatmap training preflight lost snapshotted parameters: "
            f"{missing[:5]}"
        )

    changed_tensors = 0
    changed_elements = 0
    max_abs_delta = 0.0
    delta_sq_sum = 0.0
    reference_sq_sum = 0.0

    for name, original in before.items():
        current = parameters[name].detach().to(device="cpu", copy=True)
        if not bool(torch.isfinite(current.float()).all()):
            raise RuntimeError(
                "Heatmap training preflight produced non-finite parameters: "
                f"{name}"
            )
        if current.shape != original.shape or current.dtype != original.dtype:
            raise RuntimeError(
                "Heatmap training preflight changed parameter metadata: "
                f"{name} {tuple(original.shape)}/{original.dtype} -> "
                f"{tuple(current.shape)}/{current.dtype}"
            )

        changed = current.ne(original)
        tensor_changed_elements = int(changed.sum().item())
        if tensor_changed_elements:
            changed_tensors += 1
            changed_elements += tensor_changed_elements

        delta = current.float() - original.float()
        if delta.numel():
            max_abs_delta = max(max_abs_delta, float(delta.abs().max().item()))
        delta_sq_sum += float(
            torch.sum(delta.square(), dtype=torch.float64).item()
        )
        reference_sq_sum += float(
            torch.sum(original.float().square(), dtype=torch.float64).item()
        )

    delta_l2 = math.sqrt(delta_sq_sum)
    reference_l2 = math.sqrt(reference_sq_sum)
    relative_l2 = delta_l2 / max(reference_l2, 1e-12)
    if (
        changed_elements == 0
        or not math.isfinite(delta_l2)
        or not math.isfinite(relative_l2)
        or max_abs_delta <= 0.0
    ):
        raise RuntimeError(
            "Training preflight completed an optimizer step but did not change "
            "any trainable heatmap parameters. This commonly indicates a "
            "zero/underflowed update or a disconnected loss."
        )

    return {
        "heatmap_changed_tensors": float(changed_tensors),
        "heatmap_changed_elements": float(changed_elements),
        "heatmap_param_delta_max_abs": max_abs_delta,
        "heatmap_param_delta_l2": delta_l2,
        "heatmap_param_delta_relative_l2": relative_l2,
    }


def _verify_frozen_lora_samples(
    model,
    before: dict[str, tuple[list[int], torch.Tensor]],
) -> dict[str, float]:
    if not before:
        return {}

    module = _unwrap_model(model)
    parameters = dict(module.named_parameters())
    max_abs_delta = 0.0
    sampled_elements = 0
    changed: list[str] = []

    for name, (indices, original) in before.items():
        param = parameters.get(name)
        if param is None:
            changed.append(f"{name} (missing)")
            continue
        device_indices = torch.tensor(
            indices,
            dtype=torch.long,
            device=param.device,
        )
        current = (
            param.detach()
            .reshape(-1)
            .index_select(0, device_indices)
            .to(device="cpu", copy=True)
        )
        sampled_elements += current.numel()
        delta = current.float() - original.float()
        if delta.numel():
            max_abs_delta = max(max_abs_delta, float(delta.abs().max().item()))
        if not torch.equal(current, original):
            changed.append(name)

    if changed:
        raise RuntimeError(
            "Training preflight mutated sampled values from frozen LoRA "
            f"parameters: {changed[:5]}"
        )
    return {
        "frozen_lora_sampled_tensors": float(len(before)),
        "frozen_lora_sampled_elements": float(sampled_elements),
        "frozen_lora_sample_max_abs_delta": max_abs_delta,
    }


def run_training_preflight(
    model,
    train_loader,
    optimizer,
    scheduler,
    scaler,
    cfg: dict,
    logger,
    *,
    stage_name: str,
    stage_cfg: dict,
    train_dataset,
    train_sampler=None,
    gpu_heatmap_computer=None,
    gpu_has_depth: bool = False,
    gpu_depth_normalized: bool = True,
    ema=None,
    total_train_steps: int = 1,
    dist_context=None,
    nextdit_warmup_steps: int = 0,
    l2_sp_reference: dict[str, Any] | None = None,
) -> dict[str, float]:
    """Execute one complete train batch and require a finite optimizer step.

    The optimizer/EMA mutation is process-local: the caller exits without
    saving a checkpoint after this function succeeds.
    """
    if len(train_loader) < 1:
        raise RuntimeError('Training preflight requires at least one full batch')

    preflight_epoch = 1
    if hasattr(train_dataset, 'set_epoch'):
        train_dataset.set_epoch(preflight_epoch)
    if train_sampler is not None:
        train_sampler.set_epoch(preflight_epoch)

    logger.info(
        'Running real training preflight: one batch with forward, loss, backward, '
        'DDP gradient synchronization, and an in-memory optimizer step'
    )
    static_report = assert_single_view_training_contract(
        model,
        optimizer,
        cfg,
        stage_cfg,
    )
    if static_report:
        logger.info("Single-view static safety contract passed: %s", static_report)
    strict_8gpu_smoke = smoke_audit_enabled(stage_cfg)
    strict_ppa_smoke = ppa_smoke_audit_enabled(stage_cfg)
    if strict_8gpu_smoke and strict_ppa_smoke:
        raise RuntimeError(
            "Pose-adaptation and PPA distributed smoke audits are mutually exclusive"
        )
    strict_smoke_world_size = expected_smoke_world_size() if strict_8gpu_smoke else None
    ppa_smoke_world_size = (
        expected_ppa_smoke_world_size() if strict_ppa_smoke else None
    )
    smoke_identities: list[str] = []
    smoke_providers: list[str] = []
    smoke_gradient_records: dict[str, dict[str, Any]] = {}
    smoke_gradient_handles: list[Any] = []
    actual_batch_observer = None
    if strict_8gpu_smoke:
        if (
            dist_context is None
            or not dist_context.enabled
            or dist_context.world_size != strict_smoke_world_size
        ):
            raise RuntimeError(
                "Strict pose-adaptation smoke world-size mismatch: "
                f"configured={strict_smoke_world_size} "
                f"actual={getattr(dist_context, 'world_size', None)}"
            )
        if int(cfg.get("optim", {}).get("batch_size", -1)) != EXPECTED_BATCH_PER_RANK:
            raise RuntimeError(
                "Strict pose-adaptation smoke requires batch_size=2 per rank"
            )
        def observe_actual_batch(batch_index: int, batch: dict[str, Any]) -> None:
            nonlocal smoke_identities, smoke_providers
            if smoke_identities or smoke_providers:
                raise RuntimeError(
                    "Strict pose-adaptation smoke observed more than one actual batch"
                )
            smoke_identities = [
                str(value) for value in batch.get("sample_identity", [])
            ]
            raw_provider = batch.get("history_pose_provider")
            smoke_providers = (
                [str(raw_provider)] * len(smoke_identities)
                if isinstance(raw_provider, str)
                else [str(value) for value in (raw_provider or [])]
            )
            if len(smoke_identities) != EXPECTED_BATCH_PER_RANK:
                raise RuntimeError(
                    "Strict pose-adaptation smoke actual forward batch is missing "
                    f"two sample identities: {smoke_identities}"
                )
            logger.info(
                "Strict distributed pose-adaptation actual batch: rank=%d batch=%d "
                "identities=%s provider=%s",
                dist_context.rank,
                batch_index,
                smoke_identities,
                smoke_providers,
            )

        actual_batch_observer = observe_actual_batch
        smoke_gradient_records, smoke_gradient_handles = install_gradient_hooks(model)
        logger.info(
            "Strict distributed pose-adaptation audit armed: rank=%d hooks=%d",
            dist_context.rank,
            len(smoke_gradient_handles),
        )
    elif strict_ppa_smoke:
        if (
            dist_context is None
            or not dist_context.enabled
            or dist_context.world_size != ppa_smoke_world_size
        ):
            raise RuntimeError(
                "Strict PPA smoke world-size mismatch: "
                f"configured={ppa_smoke_world_size} "
                f"actual={getattr(dist_context, 'world_size', None)}"
            )
        if int(cfg.get("optim", {}).get("batch_size", -1)) != PPA_EXPECTED_BATCH_PER_RANK:
            raise RuntimeError(
                "Strict PPA smoke requires batch_size=1 per rank"
            )

        def observe_actual_ppa_batch(
            batch_index: int,
            batch: dict[str, Any],
        ) -> None:
            nonlocal smoke_identities, smoke_providers
            if smoke_identities or smoke_providers:
                raise RuntimeError(
                    "Strict PPA smoke observed more than one actual batch"
                )
            raw_identities = batch.get("sample_identity", [])
            if isinstance(raw_identities, str):
                smoke_identities = [raw_identities]
            else:
                smoke_identities = [str(value) for value in raw_identities]
            raw_provider = batch.get("history_pose_provider")
            smoke_providers = (
                [str(raw_provider)] * len(smoke_identities)
                if isinstance(raw_provider, str)
                else [str(value) for value in (raw_provider or [])]
            )
            if len(smoke_identities) != PPA_EXPECTED_BATCH_PER_RANK:
                raise RuntimeError(
                    "Strict PPA smoke actual forward batch is missing its "
                    f"sample identity: {smoke_identities}"
                )
            logger.info(
                "Strict distributed PPA actual batch: rank=%d batch=%d "
                "identities=%s provider=%s",
                dist_context.rank,
                batch_index,
                smoke_identities,
                smoke_providers,
            )

        actual_batch_observer = observe_actual_ppa_batch
        smoke_gradient_records, smoke_gradient_handles = install_ppa_gradient_hooks(
            model
        )
        logger.info(
            "Strict distributed PPA audit armed: rank=%d hooks=%d",
            dist_context.rank,
            len(smoke_gradient_handles),
        )
    heatmap_before = _snapshot_trainable_heatmap_parameters(model, stage_cfg)
    control_gates_before = _snapshot_heatmap_control_gates(
        model,
        stage_cfg,
        cfg,
    )
    frozen_lora_before = _snapshot_frozen_lora_samples(model, stage_cfg)
    try:
        metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            scheduler,
            scaler,
            cfg,
            preflight_epoch,
            logger,
            tb_writer=None,
            global_step_offset=0,
            stage_idx=0,
            stage_name=stage_name,
            stage_cfg=stage_cfg,
            max_batches=1,
            skip_first_n_batches=None,
            vis_dir=None,
            gpu_heatmap_computer=gpu_heatmap_computer,
            gpu_has_depth=gpu_has_depth,
            gpu_depth_normalized=gpu_depth_normalized,
            ema=ema,
            metrics_jsonl_path=None,
            total_train_steps=total_train_steps,
            dist_context=dist_context,
            ckpt_manager=None,
            mid_epoch_save_every=0,
            nextdit_warmup_steps=nextdit_warmup_steps,
            l2_sp_reference=l2_sp_reference,
            actual_batch_observer=actual_batch_observer,
        )
    finally:
        for handle in smoke_gradient_handles:
            handle.remove()

    optimizer_steps = int(metrics.get('optimizer_steps', 0))
    if optimizer_steps != 1:
        raise RuntimeError(
            'Training preflight did not complete exactly one optimizer step: '
            f'optimizer_steps={optimizer_steps}'
        )

    metric_names = (
        'total_loss',
        'heatmap_loss',
        'trajectory_loss',
        'lm_loss',
        'l2_sp_loss',
    )
    invalid = {
        name: metrics.get(name)
        for name in metric_names
        if name not in metrics or not math.isfinite(float(metrics[name]))
    }
    if invalid:
        raise RuntimeError(f'Training preflight produced invalid metrics: {invalid}')
    if strict_8gpu_smoke and float(metrics['heatmap_loss']) <= 0.0:
        raise RuntimeError(
            "Strict distributed pose-adaptation smoke requires finite positive heatmap_loss"
        )
    if strict_ppa_smoke:
        required_positive = (
            "trajectory_loss",
            "heatmap_loss",
            "future_heatmap_loss",
        )
        invalid_ppa_losses = {
            name: metrics.get(name)
            for name in required_positive
            if name not in metrics
            or not math.isfinite(float(metrics[name]))
            or float(metrics[name]) <= 0.0
        }
        if invalid_ppa_losses:
            raise RuntimeError(
                "Strict PPA smoke requires positive action/history/future losses: "
                f"{invalid_ppa_losses}"
            )

    if control_gates_before and float(metrics['trajectory_loss']) <= 0.0:
        raise RuntimeError(
            "Heatmap-control preflight requires a finite, strictly positive "
            "GT trajectory flow-matching loss"
        )

    delta_metrics = _verify_trainable_heatmap_delta(model, heatmap_before)
    gate_delta_metrics = _verify_heatmap_control_gate_delta(
        model,
        control_gates_before,
    )
    frozen_lora_metrics = _verify_frozen_lora_samples(model, frozen_lora_before)
    metrics.update(delta_metrics)
    metrics.update(gate_delta_metrics)
    metrics.update(frozen_lora_metrics)

    if strict_8gpu_smoke:
        local_audit = None
        local_audit_error = None
        try:
            local_audit = build_local_rank_audit(
                model=_unwrap_model(model),
                ema=ema,
                gradient_records=smoke_gradient_records,
                identities=smoke_identities,
                providers=smoke_providers,
                optimizer_steps=optimizer_steps,
                rank=dist_context.rank,
                world_size=dist_context.world_size,
            )
        except Exception as exc:  # synchronized below; never strand peer ranks
            local_audit_error = f"{type(exc).__name__}: {exc}"
        global_audit = gather_and_validate_local_audit(
            local_audit,
            local_error=local_audit_error,
        )
        metrics["pose_adaptation_8gpu_smoke"] = global_audit
        logger.info(
            "Strict distributed pose-adaptation audit passed: unique_identities=%d "
            "hooks=%s post_digest=%s ema_digest=%s",
            global_audit["global_unique_identity_count"],
            global_audit["gradient_hook_tensors_by_rank"],
            global_audit["post_parameter_digest"][:12],
            global_audit["ema_digest"][:12],
        )
    elif strict_ppa_smoke:
        local_audit = None
        local_audit_error = None
        try:
            local_audit = build_ppa_local_rank_audit(
                model=_unwrap_model(model),
                ema=ema,
                gradient_records=smoke_gradient_records,
                identities=smoke_identities,
                providers=smoke_providers,
                optimizer_steps=optimizer_steps,
                rank=dist_context.rank,
                world_size=dist_context.world_size,
            )
        except Exception as exc:  # all ranks still enter the collective below
            local_audit_error = f"{type(exc).__name__}: {exc}"
        global_audit = gather_and_validate_ppa_local_audit(
            local_audit,
            local_error=local_audit_error,
        )
        metrics["past_plan_action_4gpu_smoke"] = global_audit
        logger.info(
            "Strict distributed PPA audit passed: unique_identities=%d "
            "hooks=%s post_digest=%s ema_digest=%s",
            global_audit["global_unique_identity_count"],
            global_audit["gradient_hook_tensors_by_rank"],
            global_audit["post_parameter_digest"][:12],
            global_audit["ema_digest"][:12],
        )

    if delta_metrics:
        logger.info(
            "Heatmap parameter delta: changed=%d tensors/%d elements "
            "max_abs=%.6e l2=%.6e relative_l2=%.6e",
            int(delta_metrics["heatmap_changed_tensors"]),
            int(delta_metrics["heatmap_changed_elements"]),
            delta_metrics["heatmap_param_delta_max_abs"],
            delta_metrics["heatmap_param_delta_l2"],
            delta_metrics["heatmap_param_delta_relative_l2"],
        )
    if gate_delta_metrics:
        logger.info(
            "Heatmap control gate delta: changed=%d gates/%d elements "
            "max_abs=%.6e",
            int(gate_delta_metrics["heatmap_control_changed_gates"]),
            int(gate_delta_metrics["heatmap_control_changed_gate_elements"]),
            gate_delta_metrics["heatmap_control_gate_delta_max_abs"],
        )
    if frozen_lora_metrics:
        logger.info(
            "Frozen LoRA integrity samples unchanged: %d tensors/%d elements",
            int(frozen_lora_metrics["frozen_lora_sampled_tensors"]),
            int(frozen_lora_metrics["frozen_lora_sampled_elements"]),
        )

    logger.info(
        'Real training preflight passed: loss=%.6f trajectory_loss=%.6f optimizer_steps=%d',
        metrics['total_loss'],
        metrics['trajectory_loss'],
        optimizer_steps,
    )
    return metrics

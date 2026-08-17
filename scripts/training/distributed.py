"""
Distributed training context and communication helpers.
"""

import logging
import os
from typing import Any

import torch
import torch.distributed as dist
import torch.nn as nn

# Submodules of NextDiTActionHead that can be listed in trainable_modules (see model_builder).
_NEXTDIT_SUBMODULE_KEYS = (
    "cond_projector",
    "traj_dit",
    "memory_encoder",
    "rgb_resampler",
    "action_encoder",
    "action_decoder",
)

SyncModuleOrParam = nn.Module | nn.Parameter

from .utils import _dist_backend, _dist_is_initialized

logger = logging.getLogger(__name__)


class DistributedContext:
    """Minimal distributed runtime context."""

    def __init__(
        self,
        enabled: bool = False,
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
        device: torch.device | None = None,
    ):
        self.enabled = enabled
        self.rank = rank
        self.local_rank = local_rank
        self.world_size = world_size
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

    @property
    def is_main(self) -> bool:
        return self.rank == 0


# ---------------------------------------------------------------------------
# Point-to-point / collective helpers
# ---------------------------------------------------------------------------

def _dist_barrier() -> None:
    if _dist_is_initialized():
        dist.barrier()


def _dist_broadcast_in_place(tensor: torch.Tensor, src: int = 0) -> torch.Tensor:
    if not _dist_is_initialized():
        return tensor
    backend = _dist_backend()
    if backend == "nccl" and not tensor.is_cuda:
        cuda_tensor = tensor.detach().to(device=torch.cuda.current_device())
        dist.broadcast(cuda_tensor, src=src)
        tensor.copy_(cuda_tensor.to(device=tensor.device, dtype=tensor.dtype))
        return tensor
    if tensor.is_cuda and backend != "nccl":
        cpu_tensor = tensor.detach().cpu()
        dist.broadcast(cpu_tensor, src=src)
        tensor.copy_(cpu_tensor.to(device=tensor.device, dtype=tensor.dtype))
        return tensor
    dist.broadcast(tensor, src=src)
    return tensor


def _dist_all_reduce_in_place(tensor: torch.Tensor) -> torch.Tensor:
    if _dist_is_initialized():
        backend = _dist_backend()
        if backend == "nccl" and not tensor.is_cuda:
            cuda_tensor = tensor.detach().to(device=torch.cuda.current_device())
            dist.all_reduce(cuda_tensor, op=dist.ReduceOp.SUM)
            tensor.copy_(cuda_tensor.to(device=tensor.device, dtype=tensor.dtype))
        elif tensor.is_cuda and backend != "nccl":
            cpu_tensor = tensor.detach().cpu()
            dist.all_reduce(cpu_tensor, op=dist.ReduceOp.SUM)
            tensor.copy_(cpu_tensor.to(device=tensor.device, dtype=tensor.dtype))
        else:
            dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
    return tensor


# ---------------------------------------------------------------------------
# Trainable-module synchronisation (gradient all-reduce for non-DDP submodules)
# ---------------------------------------------------------------------------

def _get_supported_trainable_sync_modules(
    model,
    stage_cfg: dict[str, Any],
) -> list[tuple[str, SyncModuleOrParam]]:
    trainable = set(stage_cfg.get("trainable_modules", []))
    supported_trainable = {
        "heatmap_vln",
        "lora",
        "vlm_lora",
        "llm_projector",
        "nextdit_action_head",
        "latent_queries",
        "pano_latent_adapter",
        "heatmap_tokenizer",
        "heatmap_control",
        "past_plan_action",
        *_NEXTDIT_SUBMODULE_KEYS,
    }
    unsupported = sorted(trainable - supported_trainable)
    if unsupported:
        raise RuntimeError(
            "Current multi-GPU trainable-module sync only supports trainable_modules "
            f"{sorted(supported_trainable)}. Unsupported entries: {unsupported}. "
            "Please keep other heads/backbone frozen in distributed mode, or extend "
            "scripts/training/distributed.py for new trainable groups."
        )

    sync_modules: list[tuple[str, SyncModuleOrParam]] = []

    if "llm_projector" in trainable and getattr(model, "llm_projector", None) is not None:
        if any(param.requires_grad for param in model.llm_projector.parameters()):
            sync_modules.append(("llm_projector", model.llm_projector))

    if "lora" in trainable or "vlm_lora" in trainable:
        vlm_backbone = getattr(model, "vlm_backbone", getattr(model, "qwen2_5_vl", None))
        if vlm_backbone is None:
            raise RuntimeError("LoRA is trainable but no VLM backbone was found for distributed sync.")
        lora_params = [
            param for name, param in vlm_backbone.named_parameters()
            if param.requires_grad and "lora_" in name
        ]
        if not lora_params:
            raise RuntimeError(
                "LoRA is listed in trainable_modules, but no trainable lora_ parameters "
                "were found for distributed sync. Make sure the VLM backbone is loaded "
                "before initialize_trainable_module_sync()."
            )
        sync_modules.append(("vlm_lora", nn.ParameterList(lora_params)))

    ppa_enabled = "past_plan_action" in trainable
    if "heatmap_vln" in trainable:
        if model.heatmap_vln is None:
            raise RuntimeError("heatmap_vln is trainable but has not been constructed before distributed sync.")
        explicit_head_modules = getattr(
            model.heatmap_vln,
            "trainable_head_modules",
            None,
        )
        if ppa_enabled:
            for attr_name in ("coarse", "fine"):
                module = getattr(model.heatmap_vln, attr_name, None)
                if module is not None and any(
                    parameter.requires_grad for parameter in module.parameters()
                ):
                    sync_modules.append((f"heatmap_vln.{attr_name}", module))
        elif callable(explicit_head_modules):
            child_names = {
                id(child): name
                for name, child in model.heatmap_vln.named_children()
            }
            for index, module in enumerate(explicit_head_modules()):
                if not isinstance(module, nn.Module):
                    raise RuntimeError(
                        "heatmap_vln.trainable_head_modules() must return "
                        f"nn.Module objects, got {type(module).__name__} at {index}"
                    )
                if any(param.requires_grad for param in module.parameters()):
                    attr_name = child_names.get(
                        id(module),
                        f"trainable_head_modules[{index}]",
                    )
                    sync_modules.append((f"heatmap_vln.{attr_name}", module))
        else:
            for attr_name in ["vit_dpt_fusion", "llm_dpt_fusion", "coarse", "fine"]:
                module = getattr(model.heatmap_vln, attr_name, None)
                if module is not None and any(param.requires_grad for param in module.parameters()):
                    sync_modules.append((f"heatmap_vln.{attr_name}", module))

    if "past_plan_action" in trainable:
        chain = getattr(model, "past_plan_action", None)
        if chain is None:
            raise RuntimeError(
                "past_plan_action is trainable but the model has no chain"
            )
        for attr_name in ("bridge", "future_head"):
            module = getattr(chain, attr_name, None)
            if module is not None and any(
                parameter.requires_grad for parameter in module.parameters()
            ):
                sync_modules.append((f"past_plan_action.{attr_name}", module))

    if "heatmap_tokenizer" in trainable:
        tokenizer = getattr(model, "heatmap_tokenizer", None)
        if tokenizer is None:
            raise RuntimeError("heatmap_tokenizer is trainable but missing")
        if any(param.requires_grad for param in tokenizer.parameters()):
            sync_modules.append(("heatmap_tokenizer", tokenizer))

    if "heatmap_control" in trainable:
        head = getattr(model, "nextdit_action_head", None)
        adapters_fn = getattr(head, "heatmap_control_adapters", None)
        adapters = tuple(adapters_fn()) if callable(adapters_fn) else ()
        if not adapters:
            raise RuntimeError("heatmap_control is trainable but no adapters are attached")
        for index, adapter in enumerate(adapters):
            if any(param.requires_grad for param in adapter.parameters()):
                sync_modules.append((
                    f"nextdit_action_head.heatmap_control[{index}]",
                    adapter,
                ))

    nah = getattr(model, "nextdit_action_head", None)
    if "nextdit_action_head" in trainable and nah is not None:
        if any(p.requires_grad for p in nah.parameters()):
            sync_modules.append(("nextdit_action_head", nah))
    elif nah is not None:
        for key in _NEXTDIT_SUBMODULE_KEYS:
            if key not in trainable:
                continue
            sub = getattr(nah, key, None)
            if sub is not None and any(p.requires_grad for p in sub.parameters()):
                sync_modules.append((f"nextdit_action_head.{key}", sub))

    if "latent_queries" in trainable:
        lq = getattr(model, "latent_queries", None)
        if isinstance(lq, nn.Parameter) and lq.requires_grad:
            sync_modules.append(("latent_queries", lq))

    if "pano_latent_adapter" in trainable:
        adapter = getattr(model, "pano_latent_adapter", None)
        if adapter is None:
            raise RuntimeError("pano_latent_adapter is trainable but the model has no adapter module.")
        if any(param.requires_grad for param in adapter.parameters()):
            sync_modules.append(("pano_latent_adapter", adapter))

    if not sync_modules:
        raise RuntimeError(
            "Distributed mode is enabled, but no supported trainable submodules were found for synchronization."
        )

    # This project performs manual parameter broadcast/all-reduce instead of
    # wrapping the whole pipeline in DDP.  Missing even one newly introduced
    # tensor silently lets ranks initialize and optimize different models;
    # listing a shared tensor twice corrupts the collective sequence.  Enforce
    # exact, duplicate-free coverage before the first collective.
    synchronized_ids: list[int] = []
    synchronized_names: dict[int, str] = {}
    duplicates: list[str] = []
    for sync_name, module_or_param in sync_modules:
        parameters = (
            (module_or_param,)
            if isinstance(module_or_param, nn.Parameter)
            else tuple(module_or_param.parameters())
        )
        for parameter in parameters:
            if not parameter.requires_grad:
                continue
            parameter_id = id(parameter)
            if parameter_id in synchronized_names:
                duplicates.append(
                    f"{synchronized_names[parameter_id]} and {sync_name}"
                )
            else:
                synchronized_names[parameter_id] = sync_name
            synchronized_ids.append(parameter_id)

    named_trainable = {
        id(parameter): name
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
    }
    synchronized_set = set(synchronized_ids)
    missing = sorted(
        named_trainable[parameter_id]
        for parameter_id in named_trainable.keys() - synchronized_set
    )
    extra = len(synchronized_set - named_trainable.keys())
    if duplicates or missing or extra:
        raise RuntimeError(
            "Distributed trainable-parameter sync coverage mismatch: "
            f"duplicates={duplicates[:8]}, missing={missing[:8]}, extra={extra}"
        )

    return sync_modules


def _broadcast_trainable_tensor(tensor: torch.Tensor, src: int = 0) -> None:
    if tensor.requires_grad:
        _dist_broadcast_in_place(tensor.data, src=src)


def _all_reduce_trainable_grad(param: torch.Tensor, world_size: int) -> None:
    if not param.requires_grad:
        return

    # Every rank must enqueue exactly the same collective sequence.  Heatmap
    # losses are data-dependent (for example, a local batch can contain no
    # visible target), so a trainable parameter may have ``grad is None`` on
    # one rank while another rank produced a real gradient.  Skipping the
    # all-reduce in that case shifts the collective order and eventually
    # deadlocks NCCL.  A missing local contribution is mathematically zero;
    # materialise it and participate in the same reduction on every rank.
    grad = param.grad
    if grad is None:
        grad = torch.zeros_like(
            param,
            memory_format=torch.preserve_format,
        )
        param.grad = grad
    _dist_all_reduce_in_place(grad)
    grad.div_(world_size)


def initialize_trainable_module_sync(
    model,
    stage_cfg: dict[str, Any],
    dist_context: "DistributedContext",
    logger: logging.Logger,
) -> list[tuple[str, SyncModuleOrParam]]:
    sync_modules = _get_supported_trainable_sync_modules(model, stage_cfg)
    for module_name, module_or_param in sync_modules:
        logger.info("🔄 Broadcasting trainable module: %s", module_name)
        if isinstance(module_or_param, nn.Parameter):
            _broadcast_trainable_tensor(module_or_param, src=0)
        else:
            for _, param in module_or_param.named_parameters():
                if param.requires_grad:
                    _dist_broadcast_in_place(param.data, src=0)
    _dist_barrier()
    logger.info(
        "🔗 Distributed trainable-module sync enabled for: %s",
        ", ".join(name for name, _ in sync_modules),
    )
    return sync_modules


def synchronize_trainable_module_gradients(
    sync_modules: list[tuple[str, SyncModuleOrParam]],
    dist_context: "DistributedContext",
) -> None:
    if not dist_context.enabled or dist_context.world_size <= 1:
        return
    ws = dist_context.world_size
    for _, module_or_param in sync_modules:
        if isinstance(module_or_param, nn.Parameter):
            _all_reduce_trainable_grad(module_or_param, ws)
        else:
            for param in module_or_param.parameters():
                _all_reduce_trainable_grad(param, ws)


# ---------------------------------------------------------------------------
# Init / cleanup
# ---------------------------------------------------------------------------

def init_distributed_context(cfg: dict) -> DistributedContext:
    """Initialize optional DDP runtime from config + torchrun env."""
    gpu_cfg = cfg.get("gpu", {})
    multi_gpu_cfg = gpu_cfg.get("multi_gpu", {})
    enabled = bool(multi_gpu_cfg.get("enabled", False))
    configured_devices = list(gpu_cfg.get("devices", [0]))
    world_size_env = int(os.environ.get("WORLD_SIZE", "1"))

    if not enabled:
        if world_size_env > 1:
            raise RuntimeError(
                "Detected torchrun distributed environment, but gpu.multi_gpu.enabled is false. "
                "Please enable the multi-GPU switch or launch with plain python for single-card training."
            )
        if torch.cuda.is_available():
            device_id = int(configured_devices[0]) if configured_devices else 0
            torch.cuda.set_device(device_id)
            device = torch.device(f"cuda:{device_id}")
        else:
            device = torch.device("cpu")
        return DistributedContext(enabled=False, device=device)

    if not torch.cuda.is_available():
        raise RuntimeError("Multi-GPU training requires CUDA, but CUDA is unavailable.")

    world_size = world_size_env
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    if world_size <= 1:
        requested = len(configured_devices)
        raise RuntimeError(
            "Detected gpu.multi_gpu.enabled=true but no torchrun distributed environment. "
            f"Please launch with torchrun --nproc_per_node={requested} scripts/run.py train ..."
        )

    if configured_devices and world_size != len(configured_devices):
        visible_count = torch.cuda.device_count()
        if visible_count == world_size:
            configured_devices = list(range(world_size))
            logger.warning(
                "WORLD_SIZE=%s does not match configured gpu.devices=%s; "
                "using visible CUDA device indices %s. This is expected when "
                "CUDA_VISIBLE_DEVICES is set by the launch script.",
                world_size,
                gpu_cfg.get("devices", [0]),
                configured_devices,
            )
        else:
            raise RuntimeError(
                f"WORLD_SIZE={world_size} does not match configured "
                f"gpu.devices={configured_devices}, and CUDA visible device "
                f"count is {visible_count}."
            )

    if local_rank >= len(configured_devices):
        raise RuntimeError(
            f"LOCAL_RANK={local_rank} exceeds configured gpu.devices={configured_devices}."
        )

    device_id = int(configured_devices[local_rank])
    torch.cuda.set_device(device_id)
    dist.init_process_group(
        backend=gpu_cfg.get("backend", "nccl"),
        init_method="env://",
    )
    return DistributedContext(
        enabled=True,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        device=torch.device(f"cuda:{device_id}"),
    )


def cleanup_distributed() -> None:
    if _dist_is_initialized():
        dist.destroy_process_group()

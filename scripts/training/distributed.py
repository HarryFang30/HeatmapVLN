"""
Distributed training context and communication helpers.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.distributed as dist

from .utils import _dist_is_initialized, _dist_backend

logger = logging.getLogger(__name__)


class DistributedContext:
    """Minimal distributed runtime context."""

    def __init__(
        self,
        enabled: bool = False,
        rank: int = 0,
        local_rank: int = 0,
        world_size: int = 1,
        device: Optional[torch.device] = None,
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
        if tensor.is_cuda and backend != "nccl":
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
    stage_cfg: Dict[str, Any],
) -> List[Tuple[str, nn.Module]]:
    trainable = set(stage_cfg.get("trainable_modules", []))
    supported_trainable = {"heatmap_vln", "llm_projector"}
    unsupported = sorted(trainable - supported_trainable)
    if unsupported:
        raise RuntimeError(
            "Current multi-GPU trainable-module sync only supports trainable_modules "
            f"{sorted(supported_trainable)}. Unsupported entries: {unsupported}. "
            "Please keep other heads/backbone frozen in distributed mode."
        )

    sync_modules: List[Tuple[str, nn.Module]] = []

    if "llm_projector" in trainable and getattr(model, "llm_projector", None) is not None:
        if any(param.requires_grad for param in model.llm_projector.parameters()):
            sync_modules.append(("llm_projector", model.llm_projector))

    if "heatmap_vln" in trainable:
        if model.heatmap_vln is None:
            raise RuntimeError("heatmap_vln is trainable but has not been constructed before distributed sync.")
        for attr_name in ["vit_dpt_fusion", "llm_dpt_fusion", "coarse", "fine"]:
            module = getattr(model.heatmap_vln, attr_name, None)
            if module is not None and any(param.requires_grad for param in module.parameters()):
                sync_modules.append((f"heatmap_vln.{attr_name}", module))

    if not sync_modules:
        raise RuntimeError(
            "Distributed mode is enabled, but no supported trainable submodules were found for synchronization."
        )

    return sync_modules


def initialize_trainable_module_sync(
    model,
    stage_cfg: Dict[str, Any],
    dist_context: "DistributedContext",
    logger: logging.Logger,
) -> List[Tuple[str, nn.Module]]:
    sync_modules = _get_supported_trainable_sync_modules(model, stage_cfg)
    for module_name, module in sync_modules:
        logger.info("🔄 Broadcasting trainable module: %s", module_name)
        for _, param in module.named_parameters():
            if param.requires_grad:
                _dist_broadcast_in_place(param.data, src=0)
    _dist_barrier()
    logger.info(
        "🔗 Distributed trainable-module sync enabled for: %s",
        ", ".join(name for name, _ in sync_modules),
    )
    return sync_modules


def synchronize_trainable_module_gradients(
    sync_modules: List[Tuple[str, nn.Module]],
    dist_context: "DistributedContext",
) -> None:
    if not dist_context.enabled or dist_context.world_size <= 1:
        return
    for _, module in sync_modules:
        for param in module.parameters():
            if param.requires_grad and param.grad is not None:
                _dist_all_reduce_in_place(param.grad)
                param.grad.div_(dist_context.world_size)


# ---------------------------------------------------------------------------
# Init / cleanup
# ---------------------------------------------------------------------------

def init_distributed_context(cfg: Dict) -> DistributedContext:
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
        raise RuntimeError(
            f"WORLD_SIZE={world_size} does not match configured gpu.devices={configured_devices}."
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

#!/usr/bin/env python3
"""Distributed regression smoke for rank-local ``grad=None`` handling."""

from __future__ import annotations

import os

import torch
import torch.distributed as dist
from torch import nn

from scripts.training.distributed import _all_reduce_trainable_grad


def main() -> None:
    dist.init_process_group(backend="gloo", init_method="env://")
    try:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        param = nn.Parameter(torch.tensor([1.0, -1.0]))

        # Deliberately make gradient participation differ by rank.  The old
        # implementation skipped the collective on odd ranks and deadlocked.
        if rank % 2 == 0:
            param.grad = torch.tensor(
                [float(rank + 1), float(-(rank + 1))]
            )

        _all_reduce_trainable_grad(param, world_size)

        expected_scalar = sum(
            float(other_rank + 1)
            for other_rank in range(world_size)
            if other_rank % 2 == 0
        ) / world_size
        expected = torch.tensor([expected_scalar, -expected_scalar])
        if param.grad is None or not torch.allclose(param.grad, expected):
            raise RuntimeError(
                f"rank={rank} reduced grad={param.grad}, expected={expected}"
            )
        dist.barrier()
        if rank == 0:
            print(
                "DDP_GRAD_NONE_SMOKE_OK "
                f"world_size={world_size} grad={param.grad.tolist()}"
            )
    finally:
        dist.destroy_process_group()


if __name__ == "__main__":
    # torchrun supplies these; keep the failure explicit outside torchrun.
    required = ("RANK", "WORLD_SIZE", "MASTER_ADDR", "MASTER_PORT")
    missing = [name for name in required if name not in os.environ]
    if missing:
        raise RuntimeError(f"Launch with torchrun; missing env vars: {missing}")
    main()

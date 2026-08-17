from __future__ import annotations

import os

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch import nn

from scripts.training.distributed import _all_reduce_trainable_grad


def _rank_local_missing_grad_worker(
    rank: int,
    world_size: int,
    rendezvous_file: str,
) -> None:
    # The Codex macOS runner has no resolvable host name; bind Gloo explicitly.
    os.environ.setdefault("GLOO_SOCKET_IFNAME", "lo0")
    dist.init_process_group(
        backend="gloo",
        init_method=f"file://{rendezvous_file}",
        rank=rank,
        world_size=world_size,
    )
    try:
        first = nn.Parameter(torch.tensor([1.0, -1.0]))
        second = nn.Parameter(torch.tensor([2.0, -2.0]))
        if rank == 0:
            first.grad = torch.tensor([4.0, -4.0])
        else:
            second.grad = torch.tensor([6.0, -6.0])

        # Both ranks must execute both collectives in the same order even
        # though each rank is locally missing a different gradient.
        _all_reduce_trainable_grad(first, world_size)
        _all_reduce_trainable_grad(second, world_size)

        torch.testing.assert_close(first.grad, torch.tensor([2.0, -2.0]))
        torch.testing.assert_close(second.grad, torch.tensor([3.0, -3.0]))

        with torch.no_grad():
            first.add_(first.grad, alpha=-0.1)
            second.add_(second.grad, alpha=-0.1)
        packed = torch.cat((first.detach(), second.detach()))
        gathered = [torch.empty_like(packed) for _ in range(world_size)]
        dist.all_gather(gathered, packed)
        for peer in gathered[1:]:
            torch.testing.assert_close(peer, gathered[0])
    finally:
        dist.destroy_process_group()


def test_rank_local_missing_gradients_keep_collectives_and_updates_aligned(
    tmp_path,
):
    rendezvous = tmp_path / "gloo-rendezvous"
    mp.spawn(
        _rank_local_missing_grad_worker,
        args=(2, str(rendezvous)),
        nprocs=2,
        join=True,
    )

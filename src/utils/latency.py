"""Opt-in wall-clock timing for the deployed navigation stack.

Off unless ``HEATMAPVLN_TIMING=1`` (or a caller passes ``enabled=True``). When off,
every call is a no-op: no clock reads, no CUDA synchronisation, nothing added to
payloads or responses, so served actions are byte-for-byte what they are without it.

When on, each stage boundary first waits for queued CUDA work
(``torch.cuda.synchronize``), so a stage's time is the time its GPU work took rather
than the time it took to enqueue it. Synchronising only moves the host's waiting
point; it does not change any tensor, so timing does not change behaviour either.
Give the timer the device the work runs on: ``torch.cuda.synchronize()`` without
one waits on the current device (cuda:0 unless set), not on ``--gpu_id``'s.
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import Any, Callable, Dict, Iterator, Optional

TIMING_ENV = "HEATMAPVLN_TIMING"


def timing_enabled() -> bool:
    return os.environ.get(TIMING_ENV, "").strip().lower() not in ("", "0", "false", "no", "off")


def _cuda_torch(device: Any = None) -> Any:
    """torch when CUDA is usable and ``device`` (None: the current one) is a CUDA device."""
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    if device is not None and torch.device(device).type != "cuda":
        return None
    return torch


def _cuda_sync_fn(device: Any = None) -> Callable[[], None]:
    torch = _cuda_torch(device)
    if torch is None:
        return lambda: None
    if device is None:
        return torch.cuda.synchronize
    return lambda: torch.cuda.synchronize(device)


def reset_cuda_peak(device: Any) -> None:
    """Start a new peak-memory window on ``device``; a no-op off CUDA."""
    torch = _cuda_torch(device)
    if torch is not None:
        torch.cuda.reset_peak_memory_stats(device)


def cuda_memory_mib(device: Any) -> Optional[Dict[str, float]]:
    """GPU memory in MiB, or None off CUDA.

    ``peak_allocated`` / ``peak_reserved``: this process's tensors / its caching
    allocator's pool, peak since the last ``reset_cuda_peak``.  ``device_used``: the
    whole card in use right now, every process included (CUDA contexts, a model and
    a VO server sharing the card, anyone else's jobs on it).
    """
    torch = _cuda_torch(device)
    if torch is None:
        return None
    free, total = torch.cuda.mem_get_info(device)
    return {
        "peak_allocated": round(torch.cuda.max_memory_allocated(device) / 2**20, 1),
        "peak_reserved": round(torch.cuda.max_memory_reserved(device) / 2**20, 1),
        "device_used": round((total - free) / 2**20, 1),
    }


class StageTimer:
    """Accumulates milliseconds per named stage; a repeated stage adds up and counts."""

    def __init__(
        self, enabled: Optional[bool] = None, cuda_sync: bool = True, device: Any = None
    ) -> None:
        self.enabled = timing_enabled() if enabled is None else bool(enabled)
        self.ms: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self._sync: Callable[[], None] = (
            _cuda_sync_fn(device) if (self.enabled and cuda_sync) else (lambda: None)
        )

    @contextlib.contextmanager
    def stage(self, name: str) -> Iterator[None]:
        if not self.enabled:
            yield
            return
        self._sync()
        start = time.perf_counter()
        try:
            yield
        finally:
            self._sync()
            self.add(name, (time.perf_counter() - start) * 1000.0)

    def add(self, name: str, ms: float) -> None:
        if not self.enabled:
            return
        self.ms[name] = self.ms.get(name, 0.0) + float(ms)
        self.counts[name] = self.counts.get(name, 0) + 1

    def rename(self, old: str, new: str) -> None:
        """File a finished stage under another name once its outcome is known."""
        if not self.enabled or old not in self.ms or old == new:
            return
        self.ms[new] = self.ms.get(new, 0.0) + self.ms.pop(old)
        self.counts[new] = self.counts.get(new, 0) + self.counts.pop(old)

    def as_dict(self) -> Dict[str, float]:
        return {name: round(value, 3) for name, value in self.ms.items()}

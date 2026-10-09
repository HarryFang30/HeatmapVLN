"""Opt-in wall-clock timing for the deployed navigation stack.

Off unless ``HEATMAPVLN_TIMING=1`` (or a caller passes ``enabled=True``). When off,
every call is a no-op: no clock reads, no accelerator synchronisation, nothing added
to payloads or responses, so served actions are byte-for-byte what they are without it.

When on, each stage boundary first waits for queued accelerator work
(``torch.cuda.synchronize`` / ``torch.npu.synchronize``), so a stage's time is the
time its device work took rather than the time it took to enqueue it. Synchronising
only moves the host's waiting point; it does not change any tensor, so timing does
not change behaviour either.  Give the timer the device the work runs on:
``synchronize()`` without one waits on the current device (index 0 unless set), not
on ``--gpu_id``'s.

Both NVIDIA CUDA and Ascend NPU are supported.  A device whose backend is named but
unusable raises instead of timing nothing: a silently skipped synchronisation would
measure kernel-launch time and report it as compute time, which is worse than no
number at all.  The reported field keeps its wire name ``cuda_memory_mib`` on every
backend, so client logs and ``scripts/tools/summarize_latency.py`` stay comparable
across platforms.
"""

from __future__ import annotations

import contextlib
import os
import time
from typing import Any, Callable, Dict, Iterator, Optional

TIMING_ENV = "HEATMAPVLN_TIMING"

# Accelerator backends, by torch device type.  Each exposes the synchronize /
# reset_peak_memory_stats / max_memory_allocated / max_memory_reserved / mem_get_info
# surface this module needs; torch.npu appears once torch_npu is imported.
_ACCELERATOR_TYPES = ("cuda", "npu")


def timing_enabled() -> bool:
    return os.environ.get(TIMING_ENV, "").strip().lower() not in ("", "0", "false", "no", "off")


def _backend(torch: Any, kind: str) -> Any:
    """``torch.cuda`` / ``torch.npu`` if usable, else None."""
    if kind == "npu":
        try:
            import torch_npu  # noqa: F401  (registers torch.npu)
        except ImportError:
            return None
    module = getattr(torch, kind, None)
    if module is None or not module.is_available():
        return None
    return module


def _accelerator(device: Any = None) -> Any:
    """The accelerator module ``device`` lives on, or None for CPU / no torch.

    ``device`` None means "whichever accelerator is available". Naming an
    accelerator device whose backend is unusable is an error, not a no-op.
    """
    try:
        import torch
    except ImportError:
        return None
    if device is None:
        for kind in _ACCELERATOR_TYPES:
            module = _backend(torch, kind)
            if module is not None:
                return module
        return None
    kind = torch.device(device).type
    if kind not in _ACCELERATOR_TYPES:
        return None
    module = _backend(torch, kind)
    if module is None:
        raise RuntimeError(
            f"timing was asked to synchronise {device!r}, but the {kind} backend is "
            "not usable here; refusing to report launch time as compute time"
        )
    return module


def _accel_sync_fn(device: Any = None) -> Callable[[], None]:
    backend = _accelerator(device)
    if backend is None:
        return lambda: None
    if device is None:
        return backend.synchronize
    return lambda: backend.synchronize(device)


def reset_accel_peak(device: Any) -> None:
    """Start a new peak-memory window on ``device``; a no-op on CPU."""
    backend = _accelerator(device)
    if backend is not None:
        backend.reset_peak_memory_stats(device)


def accel_memory_mib(device: Any) -> Optional[Dict[str, float]]:
    """Accelerator memory in MiB, or None on CPU.

    ``peak_allocated`` / ``peak_reserved``: this process's tensors / its caching
    allocator's pool, peak since the last ``reset_accel_peak``.  ``device_used``: the
    whole card in use right now, every process included (device contexts, a model and
    a VO server sharing the card, anyone else's jobs on it).
    """
    backend = _accelerator(device)
    if backend is None:
        return None
    free, total = backend.mem_get_info(device)
    return {
        "peak_allocated": round(backend.max_memory_allocated(device) / 2**20, 1),
        "peak_reserved": round(backend.max_memory_reserved(device) / 2**20, 1),
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
            _accel_sync_fn(device) if (self.enabled and cuda_sync) else (lambda: None)
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

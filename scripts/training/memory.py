"""
Memory management utilities for training workers and the main process.

Includes ``ShmBypassDataset`` / ``ShmBypassCollate`` — transparent wrappers
that convert tensors ↔ numpy arrays so DataLoader workers transfer data
through the regular pickle pipe instead of ``shm_open()`` (which requires
``/dev/shm``).  When ``/dev/shm`` is too small (Docker default 64 MB),
these wrappers are the only way to use ``num_workers > 0``.
"""

from __future__ import annotations

import logging
import warnings
from collections.abc import Callable
from typing import Any

import numpy as np
import torch
import torch.utils.data

logger = logging.getLogger(__name__)


_LIBC: "ctypes.CDLL" = None  # type: ignore[name-defined]


def _get_libc():
    """Lazily load and cache libc handle to avoid repeated dlopen() calls."""
    global _LIBC
    if _LIBC is None:
        try:
            import ctypes
            _LIBC = ctypes.CDLL("libc.so.6")
        except (OSError, AttributeError):
            _LIBC = False  # type: ignore[assignment]
    return _LIBC if _LIBC is not False else None


def _malloc_trim():
    """Force glibc to return freed memory to the OS."""
    try:
        libc = _get_libc()
        if libc is not None:
            libc.malloc_trim(0)
    except (OSError, AttributeError):
        pass


def _cgroup_mem_usage_gb():
    for path in (
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        "/sys/fs/cgroup/memory.current",
    ):
        try:
            with open(path) as f:
                return int(f.read().strip()) / (1024 ** 3)
        except (OSError, ValueError, FileNotFoundError):
            continue
    return -1.0


def _cgroup_mem_limit_gb():
    for path in (
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "/sys/fs/cgroup/memory.max",
    ):
        try:
            with open(path) as f:
                val = f.read().strip()
                if val == "max":
                    return -1.0
                v = int(val)
                if v > 1 << 60:
                    return -1.0
                return v / (1024 ** 3)
        except (OSError, ValueError, FileNotFoundError):
            continue
    return -1.0


_CG_LIMIT_GB = _cgroup_mem_limit_gb()


def _drop_page_cache(force: bool = False, threshold: float = 0.80):
    """Drop page cache when cgroup memory usage exceeds threshold of limit.

    Args:
        force: If True, always drop regardless of threshold.
        threshold: Fraction of cgroup limit above which to drop (default 80%).
    """
    if _CG_LIMIT_GB <= 0:
        return
    usage = _cgroup_mem_usage_gb()
    if not force and usage <= _CG_LIMIT_GB * threshold:
        return

    try:
        with open("/proc/sys/vm/drop_caches", "w") as f:
            f.write("1\n")
        after = _cgroup_mem_usage_gb()
        logger.debug(
            "[PAGE_CACHE] drop_caches: %.1fGB -> %.1fGB (limit=%.0fGB)",
            usage, after, _CG_LIMIT_GB,
        )
        return
    except PermissionError:
        pass
    except (OSError, FileNotFoundError) as e:
        logger.debug("[PAGE_CACHE] drop_caches failed: %s", e)
        return

    try:
        with open("/sys/fs/cgroup/memory/memory.force_empty", "w") as f:
            f.write("0\n")
        after = _cgroup_mem_usage_gb()
        logger.debug(
            "[PAGE_CACHE] force_empty: %.1fGB -> %.1fGB (limit=%.0fGB)",
            usage, after, _CG_LIMIT_GB,
        )
        return
    except (OSError, PermissionError):
        pass

    lbc = _get_libc()
    if lbc is not None:
        lbc.malloc_trim(0)
        logger.warning(
            "[PAGE_CACHE] cannot drop page cache (no permission). "
            "cgroup=%.1f/%.0fGB. Consider: chmod 666 /proc/sys/vm/drop_caches",
            usage, _CG_LIMIT_GB,
        )


def _worker_init_fn(worker_id):
    """Worker process init: suppress warnings + memory management."""
    import gc as _gc
    import os as _os
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", message=".*fps.*frames per second.*video metadata.*")
    warnings.filterwarnings("ignore", message="Asked to sample")

    import torch.multiprocessing as _mp
    _mp.set_sharing_strategy('file_system')

    _gc.set_threshold(700, 10, 999_999_999)

    lbc = _get_libc()
    if lbc is not None:
        try:
            lbc.mallopt(-3, 32 * 1024)   # M_MMAP_THRESHOLD  → 32 KB
            lbc.mallopt(-1, 64 * 1024)   # M_TRIM_THRESHOLD  → 64 KB
            lbc.mallopt(-8, 2)           # M_ARENA_MAX → 2
        except (OSError, AttributeError):
            pass

    try:
        if _os.environ.get("HEATMAPVLN_LOG_MEMORY", "0") != "1":
            pass
        else:
            import logging as _logging
            _wlog = _logging.getLogger("heatmapvln.worker")
            with open("/proc/self/statm", "rb") as f:
                pages = int(f.read().split()[1])
            rss_mb = pages * _os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
            _wlog.debug(
                "[WORKER init] worker_id=%d pid=%d rss=%.0fMB gc_threshold=%s",
                worker_id, _os.getpid(), rss_mb, _gc.get_threshold(),
            )
    except (OSError, ValueError, FileNotFoundError):
        pass


# ---------------------------------------------------------------------------
# /dev/shm bypass: tensor ↔ numpy conversion for DataLoader IPC
# ---------------------------------------------------------------------------

def _to_numpy(obj: Any) -> Any:
    """Recursively convert torch.Tensor → numpy.ndarray (zero-copy when
    contiguous + CPU + numpy-compatible dtype)."""
    if isinstance(obj, torch.Tensor):
        t = obj.detach().cpu()
        if t.dtype == torch.bfloat16:
            t = t.float()
        return t.numpy()
    if isinstance(obj, dict):
        return {k: _to_numpy(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_numpy(v) for v in obj)
    return obj


def _to_tensor(obj: Any) -> Any:
    """Recursively convert numpy.ndarray → torch.Tensor (zero-copy)."""
    if isinstance(obj, np.ndarray):
        return torch.from_numpy(obj)
    if isinstance(obj, dict):
        return {k: _to_tensor(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return type(obj)(_to_tensor(v) for v in obj)
    return obj


class ShmBypassDataset(torch.utils.data.Dataset):
    """Wraps any map-style dataset so ``__getitem__`` returns numpy arrays
    instead of torch tensors.  This makes the DataLoader pickle the arrays
    through the normal pipe rather than ``shm_open()``, completely
    avoiding ``/dev/shm``."""

    def __init__(self, dataset: torch.utils.data.Dataset) -> None:
        object.__setattr__(self, '_dataset', dataset)

    # -- core Dataset API --------------------------------------------------
    def __len__(self) -> int:
        return len(self._dataset)  # type: ignore[arg-type]

    def __getitem__(self, idx: int) -> Any:
        return _to_numpy(self._dataset[idx])

    # -- transparent attribute delegation ----------------------------------
    def __getattr__(self, name: str) -> Any:
        return getattr(self._dataset, name)

    def __setattr__(self, name: str, value: Any) -> None:
        setattr(self._dataset, name, value)


class ShmBypassCollate:
    """Wraps a collate function to convert numpy arrays back to tensors
    before the real collation."""

    def __init__(self, collate_fn: Callable) -> None:
        self._inner = collate_fn

    def __call__(self, batch: list[Any]) -> Any:
        batch = [_to_tensor(sample) for sample in batch]
        return self._inner(batch)

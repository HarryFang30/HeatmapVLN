"""
Memory management utilities for training workers and the main process.

Includes ``ShmBypassDataset`` / ``ShmBypassCollate`` — transparent wrappers
that convert tensors ↔ numpy arrays so DataLoader workers transfer data
through the regular pickle pipe instead of ``shm_open()`` (which requires
``/dev/shm``).  When ``/dev/shm`` is too small (Docker default 64 MB),
these wrappers are the only way to use ``num_workers > 0``.
"""

from __future__ import annotations

import sys
import os
import gc
import warnings
from typing import Any, Callable, Dict, List, Sequence, Union

import numpy as np
import torch
import torch.utils.data


def _malloc_trim():
    """Force glibc to return freed memory to the OS."""
    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
    except Exception:
        pass


def _cgroup_mem_usage_gb():
    for path in (
        "/sys/fs/cgroup/memory/memory.usage_in_bytes",
        "/sys/fs/cgroup/memory.current",
    ):
        try:
            with open(path, "r") as f:
                return int(f.read().strip()) / (1024 ** 3)
        except Exception:
            continue
    return -1.0


def _cgroup_mem_limit_gb():
    for path in (
        "/sys/fs/cgroup/memory/memory.limit_in_bytes",
        "/sys/fs/cgroup/memory.max",
    ):
        try:
            with open(path, "r") as f:
                val = f.read().strip()
                if val == "max":
                    return -1.0
                v = int(val)
                if v > 1 << 60:
                    return -1.0
                return v / (1024 ** 3)
        except Exception:
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
        print(
            f"[PAGE_CACHE] drop_caches: {usage:.1f}GB → {after:.1f}GB "
            f"(limit={_CG_LIMIT_GB:.0f}GB)",
            file=sys.stderr, flush=True,
        )
        return
    except PermissionError:
        pass
    except Exception as e:
        print(f"[PAGE_CACHE] drop_caches failed: {e}", file=sys.stderr, flush=True)
        return

    try:
        with open("/sys/fs/cgroup/memory/memory.force_empty", "w") as f:
            f.write("0\n")
        after = _cgroup_mem_usage_gb()
        print(
            f"[PAGE_CACHE] force_empty: {usage:.1f}GB → {after:.1f}GB "
            f"(limit={_CG_LIMIT_GB:.0f}GB)",
            file=sys.stderr, flush=True,
        )
        return
    except Exception:
        pass

    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.malloc_trim(0)
        print(
            f"[PAGE_CACHE] WARNING: cannot drop page cache "
            f"(no permission for drop_caches or force_empty). "
            f"cgroup={usage:.1f}/{_CG_LIMIT_GB:.0f}GB. "
            f"Consider running: chmod 666 /proc/sys/vm/drop_caches",
            file=sys.stderr, flush=True,
        )
    except Exception:
        pass


def _worker_init_fn(worker_id):
    """Worker process init: suppress warnings + memory management."""
    import gc as _gc
    import os as _os
    import sys as _sys
    import warnings
    warnings.filterwarnings("ignore")
    warnings.filterwarnings("ignore", message=".*fps.*frames per second.*video metadata.*")
    warnings.filterwarnings("ignore", message="Asked to sample")

    import torch.multiprocessing as _mp
    _mp.set_sharing_strategy('file_system')

    _gc.set_threshold(700, 10, 999_999_999)

    try:
        import ctypes
        libc = ctypes.CDLL("libc.so.6")
        libc.mallopt(-3, 32 * 1024)   # M_MMAP_THRESHOLD  → 32 KB
        libc.mallopt(-1, 64 * 1024)   # M_TRIM_THRESHOLD  → 64 KB
        libc.mallopt(-8, 2)           # M_ARENA_MAX → 2
    except Exception:
        pass

    try:
        if _os.environ.get("HEATMAPVLN_LOG_MEMORY", "0") != "1":
            pass
        else:
            with open("/proc/self/statm", "rb") as f:
                pages = int(f.read().split()[1])
            rss_mb = pages * _os.sysconf("SC_PAGE_SIZE") / (1024 * 1024)
            print(
                f"[WORKER init] worker_id={worker_id} pid={_os.getpid()} "
                f"rss={rss_mb:.0f}MB gc_threshold={_gc.get_threshold()}",
                file=_sys.stderr, flush=True,
            )
    except Exception:
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

    def __call__(self, batch: List[Any]) -> Any:
        batch = [_to_tensor(sample) for sample in batch]
        return self._inner(batch)

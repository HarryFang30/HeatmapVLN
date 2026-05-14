"""
Runtime compatibility helpers for the shared Habitat/InternNav environment.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import importlib.metadata
import json
import logging
import sys
import types
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from packaging.version import Version

LOGGER = logging.getLogger(__name__)

_HF_HUB_RELAXED_VERSION = "0.36.0"


def install_numpy_legacy_aliases() -> None:
    """Restore NumPy 1.x aliases expected by older third-party code."""
    if "float" not in np.__dict__:
        np.float = np.float64  # type: ignore[attr-defined]
    if "int" not in np.__dict__:
        np.int = np.int64  # type: ignore[attr-defined]
    if "bool" not in np.__dict__:
        np.bool = np.bool_  # type: ignore[attr-defined]


def load_model_config(model_path: str) -> dict[str, Any]:
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def detect_backbone_type(model_path: str, requested_backbone_type: str = "auto") -> str:
    if requested_backbone_type != "auto":
        if requested_backbone_type not in {"qwen2_5_vl", "qwen2_vl"}:
            raise RuntimeError(
                f"Unsupported backbone_type={requested_backbone_type}. "
                "This codebase now only supports Qwen2.5-VL."
            )
        return "qwen2_5_vl"

    cfg = load_model_config(model_path)
    model_type = cfg.get("model_type", "")
    if model_type in {"qwen2_5_vl", "qwen2_vl", "internvla_n1"}:
        return "qwen2_5_vl"
    raise RuntimeError(
        f"Unsupported model_type={model_type!r} for model_path={model_path}. "
        "This codebase now only supports Qwen2.5-VL / InternNav backbone."
    )


def _make_stub_module(name: str, attrs: dict[str, Any] | None = None) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    module.__heatmapvln_stub__ = True
    if attrs:
        for key, value in attrs.items():
            setattr(module, key, value)
    sys.modules[name] = module
    return module


@contextmanager
def _relax_huggingface_hub_version_check(logger: logging.Logger | None = None):
    """
    Transformers 4.51.0 hard-checks huggingface-hub<1.0 on import, but the
    shared environment currently ships a 1.x build that is API-compatible for
    our usage. Relax that import-time check only while importing transformers.
    """
    log = logger or LOGGER
    version_fn = importlib.metadata.version

    try:
        hub_version = version_fn("huggingface-hub")
    except importlib.metadata.PackageNotFoundError:
        hub_version = None

    if hub_version is None or Version(hub_version) < Version("1.0"):
        yield
        return

    warned = False

    def _patched_version(dist_name: str) -> str:
        nonlocal warned
        normalized = dist_name.replace("_", "-").lower()
        if normalized == "huggingface-hub":
            if not warned:
                log.warning(
                    "Relaxing import-time huggingface-hub version check: "
                    "installed=%s, reported=%s for transformers compatibility",
                    hub_version,
                    _HF_HUB_RELAXED_VERSION,
                )
                warned = True
            return _HF_HUB_RELAXED_VERSION
        return version_fn(dist_name)

    importlib.metadata.version = _patched_version
    try:
        yield
    finally:
        importlib.metadata.version = version_fn


def _is_flash_attn_stubbed() -> bool:
    module = sys.modules.get("flash_attn")
    return bool(module is not None and getattr(module, "__heatmapvln_stub__", False))


def install_flash_attn_stub(logger: logging.Logger | None = None) -> None:
    """Install a minimal flash_attn stub so transformers can import Qwen2.5-VL."""
    log = logger or LOGGER
    if _is_flash_attn_stubbed():
        return

    def _stubbed_flash_attn(*_args, **_kwargs):
        raise RuntimeError("flash_attn stub is active; use attn_implementation='sdpa' in the shared environment")

    class _FlashAttnKernelStub:
        def fwd(self, *_args, **_kwargs):
            return _stubbed_flash_attn(*_args, **_kwargs)

        def varlen_fwd(self, *_args, **_kwargs):
            return _stubbed_flash_attn(*_args, **_kwargs)

        def bwd(self, *_args, **_kwargs):
            return _stubbed_flash_attn(*_args, **_kwargs)

        def varlen_bwd(self, *_args, **_kwargs):
            return _stubbed_flash_attn(*_args, **_kwargs)

    flash_kernel_stub = _FlashAttnKernelStub()

    flash_attn_module = _make_stub_module(
        "flash_attn",
        {
            # xformers only accepts flash-attn 2.7.1-2.7.4 during import-time
            # probing. Keep the stub inside that window so optional imports don't
            # fail before we force SDPA at runtime.
            "__version__": "2.7.4",
            "flash_attn_func": _stubbed_flash_attn,
            "flash_attn_varlen_func": _stubbed_flash_attn,
        },
    )
    _make_stub_module("flash_attn_2_cuda")
    flash_attn_interface = _make_stub_module(
        "flash_attn.flash_attn_interface",
        {
            "flash_attn_func": _stubbed_flash_attn,
            "flash_attn_varlen_func": _stubbed_flash_attn,
            "flash_attn_gpu": flash_kernel_stub,
            "flash_attn_cuda": flash_kernel_stub,
        },
    )
    flash_attn_bert_padding = _make_stub_module(
        "flash_attn.bert_padding",
        {
            "index_first_axis": _stubbed_flash_attn,
            "pad_input": _stubbed_flash_attn,
            "unpad_input": _stubbed_flash_attn,
        },
    )
    flash_attn_layers = _make_stub_module("flash_attn.layers")
    flash_attn_rotary = _make_stub_module(
        "flash_attn.layers.rotary",
        {
            "apply_rotary_emb": _stubbed_flash_attn,
        },
    )
    # Some third-party libraries (notably xformers/diffusers integration paths)
    # access these stubbed modules as attributes on their parent package.
    flash_attn_module.flash_attn_interface = flash_attn_interface
    flash_attn_module.bert_padding = flash_attn_bert_padding
    flash_attn_module.layers = flash_attn_layers
    flash_attn_layers.rotary = flash_attn_rotary
    log.info("Installed flash_attn stub; the shared environment should run with SDPA instead of FlashAttention")


@dataclass(frozen=True)
class RuntimeCompatState:
    resolved_backbone_type: str
    installed_transformers_version: str
    expected_transformers_version: str | None
    flash_attn_available: bool
    flash_attn_stubbed: bool


def ensure_transformers_runtime_compat(
    model_path: str,
    requested_backbone_type: str = "auto",
    requested_attn_implementation: str = "sdpa",
    logger: logging.Logger | None = None,
) -> RuntimeCompatState:
    """Validate the active transformers stack and patch incompatible flash_attn imports."""
    log = logger or LOGGER
    install_numpy_legacy_aliases()

    resolved_backbone_type = detect_backbone_type(model_path, requested_backbone_type)
    model_cfg = load_model_config(model_path)

    with _relax_huggingface_hub_version_check(log):
        import transformers

    installed_transformers_version = transformers.__version__
    expected_transformers_version = model_cfg.get("transformers_version")

    if expected_transformers_version and Version(installed_transformers_version) != Version(expected_transformers_version):
        raise RuntimeError(
            "Transformers version mismatch for this model: "
            f"model config requires {expected_transformers_version}, "
            f"but the current environment has {installed_transformers_version}. "
            "Use the shared Habitat/InternNav baseline or switch to a matching dedicated environment."
        )

    flash_attn_available = False
    flash_attn_stubbed = False
    if _is_flash_attn_stubbed():
        flash_attn_stubbed = True
    else:
        try:
            importlib.import_module("flash_attn")
            flash_attn_available = True
        except ModuleNotFoundError:
            log.info("flash_attn is not installed in the shared environment; SDPA path remains enabled")
        except Exception as exc:
            flash_attn_stubbed = True
            install_flash_attn_stub(log)
            if requested_attn_implementation == "flash_attention_2":
                log.warning(
                    "flash_attn is unavailable (%s); the loader will avoid FlashAttention and fall back to SDPA",
                    exc,
                )
            else:
                log.info("flash_attn is unavailable in the shared environment (%s); SDPA path remains enabled", exc)

    return RuntimeCompatState(
        resolved_backbone_type=resolved_backbone_type,
        installed_transformers_version=installed_transformers_version,
        expected_transformers_version=expected_transformers_version,
        flash_attn_available=flash_attn_available,
        flash_attn_stubbed=flash_attn_stubbed,
    )

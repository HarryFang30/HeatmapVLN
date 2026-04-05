"""
Runtime compatibility helpers for the shared Habitat/InternNav environment.
"""

from __future__ import annotations

import importlib
import importlib.machinery
import json
import logging
import sys
import types
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from packaging.version import Version

LOGGER = logging.getLogger(__name__)


def install_numpy_legacy_aliases() -> None:
    """Restore NumPy 1.x aliases expected by older third-party code."""
    if "float" not in np.__dict__:
        np.float = np.float64  # type: ignore[attr-defined]
    if "int" not in np.__dict__:
        np.int = np.int64  # type: ignore[attr-defined]
    if "bool" not in np.__dict__:
        np.bool = np.bool_  # type: ignore[attr-defined]


def load_model_config(model_path: str) -> Dict[str, Any]:
    cfg_path = Path(model_path) / "config.json"
    if not cfg_path.exists():
        return {}
    with cfg_path.open("r", encoding="utf-8") as f:
        return json.load(f)


def detect_backbone_type(model_path: str, requested_backbone_type: str = "auto") -> str:
    if requested_backbone_type != "auto":
        return requested_backbone_type

    cfg = load_model_config(model_path)
    model_type = cfg.get("model_type", "")
    if model_type in {"qwen2_5_vl", "qwen2_vl"}:
        return "qwen2_5_vl"
    return "qwen3_5"


def _make_stub_module(name: str, attrs: Optional[Dict[str, Any]] = None) -> types.ModuleType:
    module = types.ModuleType(name)
    module.__spec__ = importlib.machinery.ModuleSpec(name, loader=None)
    module.__heatmapvln_stub__ = True
    if attrs:
        for key, value in attrs.items():
            setattr(module, key, value)
    sys.modules[name] = module
    return module


def _is_flash_attn_stubbed() -> bool:
    module = sys.modules.get("flash_attn")
    return bool(module is not None and getattr(module, "__heatmapvln_stub__", False))


def install_flash_attn_stub(logger: Optional[logging.Logger] = None) -> None:
    """Install a minimal flash_attn stub so transformers can import Qwen2.5-VL."""
    log = logger or LOGGER
    if _is_flash_attn_stubbed():
        return

    def _stubbed_flash_attn(*_args, **_kwargs):
        raise RuntimeError("flash_attn stub is active; use attn_implementation='sdpa' in the shared environment")

    _make_stub_module(
        "flash_attn",
        {
            "__version__": "2.8.3",
            "flash_attn_func": _stubbed_flash_attn,
            "flash_attn_varlen_func": _stubbed_flash_attn,
        },
    )
    _make_stub_module("flash_attn_2_cuda")
    _make_stub_module(
        "flash_attn.flash_attn_interface",
        {
            "flash_attn_func": _stubbed_flash_attn,
            "flash_attn_varlen_func": _stubbed_flash_attn,
        },
    )
    _make_stub_module(
        "flash_attn.bert_padding",
        {
            "index_first_axis": _stubbed_flash_attn,
            "pad_input": _stubbed_flash_attn,
            "unpad_input": _stubbed_flash_attn,
        },
    )
    _make_stub_module("flash_attn.layers")
    _make_stub_module(
        "flash_attn.layers.rotary",
        {
            "apply_rotary_emb": _stubbed_flash_attn,
        },
    )
    log.info("Installed flash_attn stub; the shared environment should run with SDPA instead of FlashAttention")


@dataclass(frozen=True)
class RuntimeCompatState:
    resolved_backbone_type: str
    installed_transformers_version: str
    expected_transformers_version: Optional[str]
    flash_attn_available: bool
    flash_attn_stubbed: bool


def ensure_transformers_runtime_compat(
    model_path: str,
    requested_backbone_type: str = "auto",
    requested_attn_implementation: str = "sdpa",
    logger: Optional[logging.Logger] = None,
) -> RuntimeCompatState:
    """Validate the active transformers stack and patch incompatible flash_attn imports."""
    log = logger or LOGGER
    install_numpy_legacy_aliases()

    resolved_backbone_type = detect_backbone_type(model_path, requested_backbone_type)
    model_cfg = load_model_config(model_path)

    import transformers

    installed_transformers_version = transformers.__version__
    expected_transformers_version = model_cfg.get("transformers_version")

    if resolved_backbone_type == "qwen3_5" and Version(installed_transformers_version) < Version("5.0.0"):
        raise RuntimeError(
            "The current shared environment is pinned to transformers 4.51.0 for Habitat/InternNav compatibility, "
            "so Qwen3.5 cannot be loaded here. Use `configs/train_config_internnav.yaml` / "
            "`models/internnav_backbone`, or switch to a dedicated Qwen3.5 environment."
        )

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

#!/usr/bin/env python3
"""Derive a partial HeatmapVLN init with a zero fine residual output."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from scripts.training.heatmap_warmstart import SPATIAL_RESET_POLICY
from scripts.training.utils import _normalize_state_key, safe_torch_load


FINE_OUTPUT_KEYS = (
    "heatmap_vln.fine.refine.4.weight",
    "heatmap_vln.fine.refine.4.bias",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_spatial_reset_state(
    source_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Keep LoRA/coarse/LLM-DPT and replace only the fine output with zeros."""
    derived: dict[str, torch.Tensor] = {}
    normalized_source = {
        _normalize_state_key(name): (name, value)
        for name, value in source_state.items()
    }
    for normalized_name, (raw_name, value) in normalized_source.items():
        if (
            "lora_" in normalized_name
            or normalized_name.startswith("heatmap_vln.llm_dpt_fusion.")
            or normalized_name.startswith("heatmap_vln.coarse.")
        ):
            derived[raw_name] = value
    for name in FINE_OUTPUT_KEYS:
        if name not in normalized_source:
            raise RuntimeError(f"Source checkpoint lacks required tensor: {name}")
        raw_name, value = normalized_source[name]
        derived[raw_name] = torch.zeros_like(value)
    return derived


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    source = args.source.resolve()
    output = args.output.resolve()
    if output.exists() and not args.force:
        raise FileExistsError(
            f"Refusing to overwrite existing derived checkpoint: {output}"
        )
    payload = safe_torch_load(str(source))
    source_state = payload.get("trainable_state_dict")
    if not isinstance(source_state, dict) or not source_state:
        raise RuntimeError("Source checkpoint has no trainable_state_dict")

    derived_state = derive_spatial_reset_state(source_state)
    output.parent.mkdir(parents=True, exist_ok=True)
    derived_payload = {
        "artifact_type": "heatmapvln_partial_warmstart",
        "metadata": {
            "created_from": str(source),
            "source_sha256": _sha256(source),
            "heatmap_warmstart_contract": {
                "policy": SPATIAL_RESET_POLICY,
                "kept_heatmap_modules": ["llm_dpt_fusion", "coarse"],
                "reset_heatmap_modules": ["vit_dpt_fusion", "fine"],
                "zero_initialized_parameters": list(FINE_OUTPUT_KEYS),
            },
        },
        "trainable_state_dict": derived_state,
    }
    torch.save(derived_payload, output)
    print(f"wrote={output}")
    print(f"tensors={len(derived_state)}")
    print(f"sha256={_sha256(output)}")


if __name__ == "__main__":
    main()


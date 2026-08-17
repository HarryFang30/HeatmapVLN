#!/usr/bin/env python3
"""Derive a fail-closed full HeatmapVLN + frozen System2 LoRA warm start."""

from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import torch

from scripts.training.heatmap_warmstart import FULL_HEAD_POLICY
from scripts.training.utils import _normalize_state_key, safe_torch_load


HEATMAP_PREFIXES = (
    "heatmap_vln.vit_dpt_fusion.",
    "heatmap_vln.llm_dpt_fusion.",
    "heatmap_vln.coarse.",
    "heatmap_vln.fine.",
)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def derive_full_head_state(
    source_state: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Keep exactly the System2 LoRA tensors and complete heatmap head."""
    derived: dict[str, torch.Tensor] = {}
    normalized_seen: set[str] = set()
    for raw_name, value in source_state.items():
        name = _normalize_state_key(raw_name)
        if name in normalized_seen:
            raise RuntimeError(f"Duplicate normalized source key: {name}")
        normalized_seen.add(name)
        if "lora_" in name or name.startswith(HEATMAP_PREFIXES):
            derived[raw_name] = value
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

    derived_state = derive_full_head_state(source_state)
    output.parent.mkdir(parents=True, exist_ok=True)
    derived_payload = {
        "artifact_type": "heatmapvln_full_head_warmstart",
        "metadata": {
            "created_from": str(source),
            "source_sha256": _sha256(source),
            "heatmap_warmstart_contract": {
                "policy": FULL_HEAD_POLICY,
                "kept_heatmap_modules": [
                    "vit_dpt_fusion",
                    "llm_dpt_fusion",
                    "coarse",
                    "fine",
                ],
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

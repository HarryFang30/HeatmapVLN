#!/usr/bin/env python3
"""Create the provenance-locked 53-tensor single-view heatmap initializer."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.training.single_view_heatmap_warmstart import (  # noqa: E402
    build_artifact,
    file_sha256,
    save_artifact_exclusive,
)
from src.models.heatmap import SingleViewFourDirectionHeatmapHead  # noqa: E402


def _load_source(path: Path):
    import torch

    try:
        payload = torch.load(path, map_location="cpu", weights_only=True)
    except TypeError as exc:
        raise RuntimeError(
            "This migration requires torch.load(..., weights_only=True)"
        ) from exc
    if not isinstance(payload, dict) or not isinstance(
        payload.get("trainable_state_dict"), dict
    ):
        raise RuntimeError("source checkpoint has no trainable_state_dict")
    return payload["trainable_state_dict"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_path = args.source.resolve(strict=True)
    config_path = args.config.resolve(strict=True)
    cfg = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    heatmap_cfg = cfg.get("model", {}).get("heatmap", {})
    if heatmap_cfg.get("input_mode") != "internnav_single_view":
        raise RuntimeError("config is not internnav_single_view")
    if cfg.get("model", {}).get("llm", {}).get("use_lora") is not False:
        raise RuntimeError("initializer config must explicitly disable LoRA")
    traj_cfg = heatmap_cfg.get("trajectory", {})
    head = SingleViewFourDirectionHeatmapHead(
        c_vit=int(heatmap_cfg.get("c_vit", 1280)),
        c_merged=int(heatmap_cfg.get("c_llm", 3584)),
        c_fused=int(heatmap_cfg.get("c_fused", 256)),
        vit_layer_indices=tuple(
            heatmap_cfg.get("vit_layer_indices", (7, 15, 23, 31))
        ),
        trajectory_num_freqs=int(traj_cfg.get("num_freqs", 16)),
        trajectory_num_heads=int(traj_cfg.get("num_heads", 4)),
        trajectory_num_layers=int(traj_cfg.get("num_layers", 2)),
        max_spatial_range=float(traj_cfg.get("max_spatial_range", 10.0)),
        coarse_logit_residual=bool(
            heatmap_cfg.get("coarse_logit_residual", False)
        ),
        joint_panorama_inference=bool(
            heatmap_cfg.get("joint_panorama_inference", True)
        ),
    )
    target_state = {
        f"heatmap_vln.{name}": value
        for name, value in head.state_dict().items()
    }
    artifact = build_artifact(
        _load_source(source_path),
        target_state,
        source_checkpoint=str(source_path),
        source_checkpoint_sha256=file_sha256(source_path),
        enforce_audited_source=True,
    )
    output = save_artifact_exclusive(artifact, args.output)
    print(f"wrote={output.resolve()}")
    print(f"tensor_count={len(artifact['trainable_state_dict'])}")
    print(f"artifact_sha256={file_sha256(output)}")
    print(
        "selected_state_content_sha256="
        + artifact["metadata"]["selection_contract"][
            "selected_state_content_sha256"
        ]
    )


if __name__ == "__main__":
    main()

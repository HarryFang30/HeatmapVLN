"""
Convert InternNav (InternVLA-N1) model into two parts:
  1. A standard Qwen2.5-VL backbone checkpoint  (models/internnav_backbone/)
  2. A separate System 1 checkpoint             (models/internnav_system1.safetensors)

Usage:
    python scripts/tools/convert_internnav_backbone.py \
        --src /workspace/InternNav_Model \
        --backbone-dst models/internnav_backbone \
        --system1-dst models/internnav_system1.safetensors

The script strips the InternNav-specific System 1 branch and preserves the
native Qwen2.5-VL backbone parameter names expected by the shared
transformers 4.51.0 baseline:

    InternNav key             ->  Qwen2.5-VL key
    ─────────────────────────────────────────────
    visual.*                  ->  visual.*
    model.embed_tokens.*      ->  model.embed_tokens.*
    model.layers.*            ->  model.layers.*
    model.norm.*              ->  model.norm.*
    lm_head.*                 ->  lm_head.*  (unchanged)

System 1 weights (model.latent_queries, model.cond_projector.*,
model.traj_dit.*, model.memory_encoder.*, model.rgb_model.*,
model.rgb_resampler.*, model.action_encoder.*, model.action_decoder.*)
are stripped of the 'model.' prefix and saved separately.
"""

import argparse
import json
import logging
import os
import shutil
from collections import OrderedDict
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

SYSTEM1_PREFIXES = (
    "model.latent_queries",
    "model.cond_projector.",
    "model.traj_dit.",
    "model.memory_encoder.",
    "model.rgb_model.",
    "model.rgb_resampler.",
    "model.action_encoder.",
    "model.action_decoder.",
)

TOKENIZER_FILES = [
    "tokenizer_config.json",
    "special_tokens_map.json",
    "added_tokens.json",
    "vocab.json",
    "merges.txt",
    "chat_template.json",
    "preprocessor_config.json",
    "generation_config.json",
]


def _is_system1_key(key: str) -> bool:
    return any(key.startswith(p) or key == p.rstrip(".") for p in SYSTEM1_PREFIXES)


def _remap_backbone_key(key: str) -> str:
    """Remap InternNav backbone key -> Qwen2.5-VL key."""
    # InternNav already stores the backbone using the native naming scheme
    # expected by Qwen2.5-VL in transformers 4.51.0. Keep those keys intact
    # and only separate the System 1 branch below.
    return key


def _remap_system1_key(key: str) -> str:
    """Strip 'model.' prefix from System 1 keys for NextDiTActionHead."""
    if key.startswith("model."):
        return key[len("model."):]
    return key


def _build_qwen25vl_config(src_config: dict) -> dict:
    """Create a Qwen2.5-VL compatible config.json from InternNav config."""
    cfg = dict(src_config)
    cfg["architectures"] = ["Qwen2_5_VLForConditionalGeneration"]
    cfg["model_type"] = "qwen2_5_vl"
    cfg["auto_map"] = {
        "AutoConfig": "transformers.Qwen2_5_VLConfig",
        "AutoModelForCausalLM": "transformers.Qwen2_5_VLForConditionalGeneration",
    }
    for k in ("n_query", "system1", "model_cfg"):
        cfg.pop(k, None)
    return cfg


def convert(
    src_dir: str,
    backbone_dst: str,
    system1_dst: str,
    max_shard_size_gb: float = 5.0,
):
    src_path = Path(src_dir)
    backbone_path = Path(backbone_dst)
    system1_path = Path(system1_dst)

    with open(src_path / "config.json") as f:
        src_config = json.load(f)

    with open(src_path / "model.safetensors.index.json") as f:
        src_index = json.load(f)
    weight_map = src_index["weight_map"]

    shards = sorted(set(weight_map.values()))
    logger.info("Source shards: %s", shards)

    backbone_path.mkdir(parents=True, exist_ok=True)

    qwen_config = _build_qwen25vl_config(src_config)
    with open(backbone_path / "config.json", "w") as f:
        json.dump(qwen_config, f, indent=2, ensure_ascii=False)
    logger.info("Wrote %s", backbone_path / "config.json")

    for fname in TOKENIZER_FILES:
        src_file = src_path / fname
        if src_file.exists():
            shutil.copy2(src_file, backbone_path / fname)
            logger.info("Copied %s", fname)

    backbone_tensors: OrderedDict = OrderedDict()
    system1_tensors: OrderedDict = OrderedDict()

    for shard_name in shards:
        shard_file = src_path / shard_name
        if not shard_file.exists():
            cache_file = src_path / ".cache" / "huggingface" / "download" / shard_name
            if cache_file.exists():
                shard_file = cache_file
            else:
                logger.warning("Shard %s not found, skipping", shard_name)
                continue

        logger.info("Processing shard: %s", shard_file)
        with safe_open(str(shard_file), framework="pt", device="cpu") as f:
            for key in f.keys():
                tensor = f.get_tensor(key)
                if _is_system1_key(key):
                    new_key = _remap_system1_key(key)
                    system1_tensors[new_key] = tensor
                    logger.info("  S1  %-50s -> %s  %s", key, new_key, tuple(tensor.shape))
                else:
                    new_key = _remap_backbone_key(key)
                    backbone_tensors[new_key] = tensor
                    if new_key != key:
                        logger.info("  BB  %-50s -> %s", key, new_key)

    logger.info(
        "Backbone: %d tensors, System1: %d tensors",
        len(backbone_tensors), len(system1_tensors),
    )

    max_shard_bytes = int(max_shard_size_gb * 1e9)
    current_shard: OrderedDict = OrderedDict()
    current_size = 0
    shard_idx = 1
    bb_weight_map: dict = {}
    shard_files = []

    def _flush_shard():
        nonlocal current_shard, current_size, shard_idx
        if not current_shard:
            return
        shard_name = f"model-{shard_idx:05d}-of-PLACEHOLDER.safetensors"
        out_path = backbone_path / shard_name
        save_file(current_shard, str(out_path))
        shard_files.append(shard_name)
        for k in current_shard:
            bb_weight_map[k] = shard_name
        logger.info("  Wrote shard %s (%d tensors)", shard_name, len(current_shard))
        shard_idx += 1
        current_shard = OrderedDict()
        current_size = 0

    for key, tensor in backbone_tensors.items():
        tensor_bytes = tensor.numel() * tensor.element_size()
        if current_size + tensor_bytes > max_shard_bytes and current_shard:
            _flush_shard()
        current_shard[key] = tensor
        current_size += tensor_bytes
    _flush_shard()

    total_shards = len(shard_files)
    renamed_map: dict = {}
    for old_name in shard_files:
        new_name = old_name.replace("PLACEHOLDER", f"{total_shards:05d}")
        old_path = backbone_path / old_name
        new_path = backbone_path / new_name
        old_path.rename(new_path)
        for k, v in bb_weight_map.items():
            if v == old_name:
                renamed_map[k] = new_name
    bb_weight_map = renamed_map

    total_size = sum(t.numel() * t.element_size() for t in backbone_tensors.values())
    index = {
        "metadata": {"total_size": total_size},
        "weight_map": bb_weight_map,
    }
    with open(backbone_path / "model.safetensors.index.json", "w") as f:
        json.dump(index, f, indent=2)
    logger.info("Wrote backbone index (%d keys, %.2f GB)", len(bb_weight_map), total_size / 1e9)

    system1_path.parent.mkdir(parents=True, exist_ok=True)
    save_file(system1_tensors, str(system1_path))
    s1_size = sum(t.numel() * t.element_size() for t in system1_tensors.values())
    logger.info("Wrote System 1 weights: %s (%d tensors, %.2f GB)", system1_path, len(system1_tensors), s1_size / 1e9)

    logger.info("Done! Backbone: %s, System1: %s", backbone_path, system1_path)
    logger.info("System 1 keys: %s", list(system1_tensors.keys()))


def main():
    parser = argparse.ArgumentParser(description="Convert InternNav model for HeatmapVLN")
    parser.add_argument("--src", default="/workspace/InternNav_Model", help="InternNav model directory")
    parser.add_argument("--backbone-dst", default="models/internnav_backbone", help="Output backbone directory")
    parser.add_argument("--system1-dst", default="models/internnav_system1.safetensors", help="Output System 1 weights")
    parser.add_argument("--max-shard-size-gb", type=float, default=5.0, help="Max shard size in GB")
    args = parser.parse_args()
    convert(args.src, args.backbone_dst, args.system1_dst, args.max_shard_size_gb)


if __name__ == "__main__":
    main()

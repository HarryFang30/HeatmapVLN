"""
Shared utility functions used across multiple training modules.
"""

import logging
import math
import yaml
import torch
import torch.nn as nn
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributed helpers (used by distributed.py, train_loop.py, etc.)
# ---------------------------------------------------------------------------

def _dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _dist_backend() -> Optional[str]:
    return dist.get_backend() if _dist_is_initialized() else None


# ---------------------------------------------------------------------------
# State-dict normalisation (handles DDP "module." prefix)
# ---------------------------------------------------------------------------

def _normalize_state_key(name: str) -> str:
    if name.startswith("module."):
        name = name[len("module."):]
    return name.replace(".module.", ".")


def _normalized_model_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    return {
        _normalize_state_key(name): value
        for name, value in model.state_dict().items()
    }


def _normalized_trainable_param_names(model: nn.Module) -> set[str]:
    return {
        _normalize_state_key(name)
        for name, param in model.named_parameters()
        if param.requires_grad
    }


def _load_normalized_state_dict(
    model: nn.Module,
    state_dict: Dict[str, torch.Tensor],
) -> Tuple[List[str], List[str], int]:
    current_state = model.state_dict()
    normalized_to_actual = {
        _normalize_state_key(name): name
        for name in current_state.keys()
    }
    remapped_state_dict = {}
    skipped_shape = []
    for name, value in state_dict.items():
        actual_name = normalized_to_actual.get(_normalize_state_key(name))
        if actual_name is not None:
            if current_state[actual_name].shape != value.shape:
                skipped_shape.append(
                    f"{actual_name}: ckpt {tuple(value.shape)} vs model {tuple(current_state[actual_name].shape)}"
                )
                continue
            remapped_state_dict[actual_name] = value
    if skipped_shape:
        logger.warning(
            "Skipped %d params due to shape mismatch:\n  %s",
            len(skipped_shape), "\n  ".join(skipped_shape),
        )
    missing, unexpected = model.load_state_dict(remapped_state_dict, strict=False)
    return missing, unexpected, len(remapped_state_dict)


# ---------------------------------------------------------------------------
# Trainable-param helpers
# ---------------------------------------------------------------------------

def _get_trainable_params(model: nn.Module) -> List[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Timing / formatting helpers
# ---------------------------------------------------------------------------

def _mean_timing(stats: Dict[str, float], count: int, key: str) -> float:
    if count <= 0:
        return 0.0
    return stats.get(key, 0.0) / count


def _format_qwen_internal_timing(stats: Dict[str, float], count: int) -> str:
    if count <= 0:
        return ""

    def avg(key: str) -> float:
        return _mean_timing(stats, count, key)

    qwen_vis = avg('qwen_visual_encode_s')
    qwen_lm = avg('qwen_language_model_s')
    qwen_layers = avg('qwen_llm_layers_s')
    qwen_full = avg('qwen_llm_full_attn_s')
    qwen_linear = avg('qwen_llm_linear_attn_s')
    qwen_mlp = avg('qwen_llm_mlp_s')
    qwen_norm = avg('qwen_llm_norm_s')
    qwen_patch = avg('qwen_visual_patch_embed_s')
    qwen_pos = avg('qwen_visual_pos_embed_s')
    qwen_rot = avg('qwen_visual_rotary_s')
    qwen_blocks = avg('qwen_visual_blocks_s')
    qwen_attn = avg('qwen_visual_attn_s')
    qwen_vmlp = avg('qwen_visual_mlp_s')
    qwen_vnorm = avg('qwen_visual_norm_s')
    qwen_merger = avg('qwen_visual_merger_s')

    sections = []
    if any(v > 0 for v in [qwen_vis, qwen_lm, qwen_layers, qwen_full, qwen_linear, qwen_mlp, qwen_norm]):
        qwen_nonlayer = max(qwen_lm - qwen_layers, 0.0)
        qwen_lres = max(qwen_layers - qwen_full - qwen_linear - qwen_mlp - qwen_norm, 0.0)
        qwen_residual = max(avg('qwen_forward_s') - qwen_vis - qwen_lm, 0.0)
        sections.append(
            f"Q[s] vis={qwen_vis:.3f} lm={qwen_lm:.3f} layers={qwen_layers:.3f} "
            f"full={qwen_full:.3f} linear={qwen_linear:.3f} mlp={qwen_mlp:.3f} "
            f"norm={qwen_norm:.3f} lres={qwen_lres:.3f} nonlayer={qwen_nonlayer:.3f} "
            f"residual={qwen_residual:.3f}"
        )

    if any(v > 0 for v in [qwen_patch, qwen_pos, qwen_rot, qwen_blocks, qwen_attn, qwen_vmlp, qwen_vnorm, qwen_merger]):
        qwen_vres = max(qwen_blocks - qwen_attn - qwen_vmlp - qwen_vnorm, 0.0)
        qwen_vnon = max(qwen_vis - qwen_patch - qwen_pos - qwen_rot - qwen_blocks - qwen_merger, 0.0)
        sections.append(
            f"QV[s] patch={qwen_patch:.3f} pos={qwen_pos:.3f} rot={qwen_rot:.3f} "
            f"blocks={qwen_blocks:.3f} attn={qwen_attn:.3f} mlp={qwen_vmlp:.3f} "
            f"norm={qwen_vnorm:.3f} merger={qwen_merger:.3f} vres={qwen_vres:.3f} "
            f"vnon={qwen_vnon:.3f}"
        )

    return " | ".join(sections)


def _format_decode_internal_timing(stats: Dict[str, float], count: int) -> str:
    if count <= 0:
        return ""

    def avg(key: str) -> float:
        return _mean_timing(stats, count, key)

    vit = avg('decode_vit_fusion_s')
    llm = avg('decode_llm_fusion_s')
    coarse = avg('decode_coarse_s')
    fine = avg('decode_fine_s')
    post = avg('decode_post_s')
    if not any(v > 0 for v in [vit, llm, coarse, fine, post]):
        return ""
    return (
        f"D[s] vit={vit:.3f} llm={llm:.3f} coarse={coarse:.3f} "
        f"fine={fine:.3f} post={post:.3f}"
    )


# ---------------------------------------------------------------------------
# Config / seed
# ---------------------------------------------------------------------------

def load_config(config_path: str) -> Dict:
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random
    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

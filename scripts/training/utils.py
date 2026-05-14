"""
Shared utility functions used across multiple training modules.
"""

import logging
from contextlib import nullcontext

import torch
import torch.distributed as dist
import torch.nn as nn
import yaml
from torch.nn.parallel import DistributedDataParallel as DDP

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Distributed helpers (used by distributed.py, train_loop.py, etc.)
# ---------------------------------------------------------------------------

def _dist_is_initialized() -> bool:
    return dist.is_available() and dist.is_initialized()


def _unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, DDP) else model


def _dist_backend() -> str | None:
    return dist.get_backend() if _dist_is_initialized() else None


# ---------------------------------------------------------------------------
# State-dict normalisation (handles DDP "module." prefix)
# ---------------------------------------------------------------------------

def _normalize_state_key(name: str) -> str:
    if name.startswith("module."):
        name = name[len("module."):]
    name = name.replace(".module.", ".")

    # Older HeatmapVLN checkpoints were saved before the backbone wrapper was
    # renamed from qwen3_5 to qwen2_5_vl.  Normalise the prefix so Stage 1 LoRA
    # weights remain loadable when used as init weights for newer runs.
    prefix_aliases = {
        "qwen3_5.": "qwen2_5_vl.",
        "qwen3_5_vl.": "qwen2_5_vl.",
    }
    for old_prefix, new_prefix in prefix_aliases.items():
        if name.startswith(old_prefix):
            return new_prefix + name[len(old_prefix):]

    return name


def _normalized_model_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
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
    state_dict: dict[str, torch.Tensor],
) -> tuple[list[str], list[str], int]:
    current_state = model.state_dict()
    normalized_to_actual = {
        _normalize_state_key(name): name
        for name in current_state
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

def _get_trainable_params(model: nn.Module) -> list[torch.nn.Parameter]:
    return [p for p in model.parameters() if p.requires_grad]


# ---------------------------------------------------------------------------
# Timing / formatting helpers
# ---------------------------------------------------------------------------

def _mean_timing(stats: dict[str, float], count: int, key: str) -> float:
    if count <= 0:
        return 0.0
    return stats.get(key, 0.0) / count


def _format_qwen_internal_timing(stats: dict[str, float], count: int) -> str:
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


def _format_decode_internal_timing(stats: dict[str, float], count: int) -> str:
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

def resolve_amp_dtype(amp_type: str | None) -> torch.dtype | None:
    """Map config AMP strings to torch dtypes."""
    normalized = "bf16" if amp_type is None else str(amp_type).lower()
    amp_dtypes = {
        "bf16": torch.bfloat16,
        "bfloat16": torch.bfloat16,
        "fp16": torch.float16,
        "float16": torch.float16,
        "none": None,
        "off": None,
        "false": None,
        "disabled": None,
    }
    if normalized not in amp_dtypes:
        raise ValueError(f"Unsupported AMP mode: {amp_type!r}")
    return amp_dtypes[normalized]


def make_autocast_context(
    device: torch.device | str,
    amp_type: str | None = "bf16",
):
    """Return a no-op or autocast context based on runtime device + AMP mode."""
    resolved_device = torch.device(device)
    dtype = resolve_amp_dtype(amp_type)
    if resolved_device.type != "cuda" or dtype is None:
        return nullcontext()
    return torch.autocast(device_type=resolved_device.type, dtype=dtype)


def make_grad_scaler(
    device: torch.device | str,
    amp_type: str | None = "bf16",
):
    """Build a CUDA GradScaler only when fp16 AMP is actually active."""
    resolved_device = torch.device(device)
    if resolved_device.type != "cuda" or resolve_amp_dtype(amp_type) != torch.float16:
        return None
    from torch.amp import GradScaler

    return GradScaler(resolved_device.type)


def load_config(config_path: str, validate: bool = True) -> dict:
    """Load a YAML config file, optionally validating against the schema.

    When *validate* is True (default), typos and type errors in the
    config are caught immediately at startup instead of causing cryptic
    ``KeyError`` deep in the training loop, and schema defaults are
    materialized into the returned dict.
    """
    with open(config_path) as f:
        cfg = yaml.safe_load(f)
    if validate:
        from src.config_schema import normalize_config

        return normalize_config(cfg)
    from src.config_schema import prepare_config_for_use

    return prepare_config_for_use(cfg)


def safe_torch_load(
    path: str,
    map_location: str = "cpu",
) -> dict:
    """Load a checkpoint with consistent safety settings.

    Uses ``weights_only=False`` because our checkpoints contain optimizer
    state dicts and other non-tensor objects.  Only load checkpoints you
    trust (i.e. produced by this project).
    """
    return torch.load(path, map_location=map_location, weights_only=False)


def build_heatmap_loss_fn(
    cfg: dict,
    device: torch.device,
    temperature: float | None = None,
    lambda_neg_override: float | None = None,
) -> "HeatmapVLNLoss":
    """Centralized factory for HeatmapVLNLoss — avoids duplication across train/validate."""
    from src.models.heatmap import HeatmapVLNLoss

    hm_cfg = cfg.get('loss', {}).get('heatmap_vln', {})
    return HeatmapVLNLoss(
        lambda_vis=hm_cfg.get('lambda_vis', 1.0),
        lambda_coord=hm_cfg.get('lambda_coord', 1.0),
        lambda_kl=hm_cfg.get('lambda_kl', hm_cfg.get('lambda_pos', 1.0)),
        lambda_peak=hm_cfg.get('lambda_peak', 1.0),
        lambda_neg=lambda_neg_override if lambda_neg_override is not None else hm_cfg.get('lambda_neg', 0.0),
        temperature=temperature if temperature is not None else hm_cfg.get('temperature', 1.0),
        heatmap_size=tuple(cfg['model'].get('heatmap', {}).get('heatmap_size', cfg['data']['init_hm_size'])),
        vis_pos_weight=hm_cfg.get('vis_pos_weight', 1.0),
    ).to(device)


def set_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    import random

    import numpy as np
    random.seed(seed)
    np.random.seed(seed)

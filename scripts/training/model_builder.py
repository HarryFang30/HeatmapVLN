"""
Model construction and freeze/unfreeze strategies.
"""

import logging
from pathlib import Path

import torch.nn as nn

from src.models.lora_utils import resolve_lora_layer_indices
from src.models.pipeline import VLNPipeline, VLNPipelineConfig
from src.models.runtime_compat import ensure_transformers_runtime_compat

logger = logging.getLogger(__name__)


def build_model(
    cfg: dict,
    verbose: bool = True,
    device: str | None = None,
    enable_action_head: bool | None = None,
) -> VLNPipeline:
    """Build the VLN Pipeline from a config dict.

    Args:
        cfg: Full training/eval config dictionary.
        verbose: Whether to log model construction details.
        device: Override ``cfg['model']['device']`` (useful for eval scripts).
        enable_action_head: Override ``cfg['model']['action_head']['enable']``.
    """
    model_cfg = cfg['model']
    llm_cfg = model_cfg.get('llm', {})
    heatmap_cfg = model_cfg.get('heatmap', {})
    action_cfg = model_cfg.get('action_head', {})
    nextdit_cfg = action_cfg.get('nextdit', {})
    resolved_lora_layers = resolve_lora_layer_indices(llm_cfg, heatmap_cfg, logger=logger)
    llm_model_path = llm_cfg.get('model_path', './models/internnav_backbone')

    ensure_transformers_runtime_compat(
        model_path=llm_model_path,
        requested_backbone_type=llm_cfg.get('backbone_type', 'qwen2_5_vl'),
        requested_attn_implementation=llm_cfg.get('attn_implementation', 'sdpa'),
        logger=logger,
    )

    effective_device = device if device is not None else model_cfg.get('device', 'cuda')
    effective_action_head = enable_action_head if enable_action_head is not None else action_cfg.get('enable', True)

    config = VLNPipelineConfig(
        llm_model_path=llm_model_path,
        llm_backbone_type=llm_cfg.get('backbone_type', 'qwen2_5_vl'),
        llm_hidden_dim=llm_cfg.get('hidden_dim', 3584),
        llm_token_dim=llm_cfg.get('token_dim', 896),
        llm_torch_dtype=llm_cfg.get('torch_dtype', 'bfloat16'),
        llm_attn_implementation=llm_cfg.get('attn_implementation', 'sdpa'),
        max_video_frames=llm_cfg.get('max_video_frames', 16),
        llm_gradient_checkpointing=llm_cfg.get('gradient_checkpointing', False),
        llm_enable_internal_profiling=llm_cfg.get('enable_internal_profiling', False),
        enable_runtime_timing=cfg.get('log', {}).get('enable_timing', False),
        llm_enable_compile=llm_cfg.get('enable_compile', False),
        llm_compile_mode=llm_cfg.get('compile_mode', 'reduce-overhead'),
        llm_compile_backend=llm_cfg.get('compile_backend', 'inductor'),

        enable_packing=llm_cfg.get('enable_packing', False),
        max_seq_length=llm_cfg.get('max_seq_length', 4096),
        spatial_merge_size=llm_cfg.get('spatial_merge_size', 2),

        internnav_system1_path=nextdit_cfg.get('internnav_system1_path', ''),
        internnav_model_path=nextdit_cfg.get('internnav_model_path', ''),

        device=effective_device,

        enable_heatmap=heatmap_cfg.get('enable', True),
        heatmap_c_vit=heatmap_cfg.get('c_vit', 1280),
        heatmap_c_llm=heatmap_cfg.get('c_llm', 3584),
        heatmap_c_fused=heatmap_cfg.get('c_fused', 256),
        heatmap_vit_layer_indices=heatmap_cfg.get('vit_layer_indices', [7, 15, 23, 31]),
        heatmap_llm_layer_indices=heatmap_cfg.get('llm_layer_indices', [6, 13, 20]),
        heatmap_size=tuple(heatmap_cfg.get('heatmap_size', cfg['data']['init_hm_size'])),
        image_size=heatmap_cfg.get('image_size', cfg['data']['image_size'][0]),
        heatmap_lambda_vis=heatmap_cfg.get('lambda_vis', 1.0),
        heatmap_lambda_coord=heatmap_cfg.get('lambda_coord', 1.0),
        heatmap_lambda_kl=heatmap_cfg.get('lambda_kl', heatmap_cfg.get('lambda_pos', 1.0)),
        heatmap_lambda_peak=heatmap_cfg.get('lambda_peak', 1.0),
        heatmap_trajectory_config=heatmap_cfg.get('trajectory', None),

        use_lora=llm_cfg.get('use_lora', False),
        lora_rank=llm_cfg.get('lora_rank', 16),
        lora_alpha=llm_cfg.get('lora_alpha', 32),
        lora_num_layers=llm_cfg.get('lora_num_layers', 4),
        lora_layer_indices=resolved_lora_layers,
        lora_dropout=llm_cfg.get('lora_dropout', 0.05),
        lora_target_modules=llm_cfg.get('lora_target_modules', None),
        heatmap_trains_backbone=heatmap_cfg.get('heatmap_trains_backbone', False),

        enable_action_head=effective_action_head,

        nextdit_enabled=nextdit_cfg.get('enabled', False),
        nextdit_vlm_hidden_dim=nextdit_cfg.get('vlm_hidden_dim', 3584),
        nextdit_latent_emb_size=nextdit_cfg.get('latent_emb_size', 768),
        nextdit_n_query=nextdit_cfg.get('n_query', 4),
        nextdit_dit_dim=nextdit_cfg.get('dit_dim', 384),
        nextdit_dit_layers=nextdit_cfg.get('dit_layers', 12),
        nextdit_dit_heads=nextdit_cfg.get('dit_heads', 6),
        nextdit_dit_kv_heads=nextdit_cfg.get('dit_kv_heads', 6),
        nextdit_dit_ffn_dim_multiplier=nextdit_cfg.get('dit_ffn_dim_multiplier', 2 / 3),
        nextdit_predict_steps=nextdit_cfg.get('predict_steps', 32),
        nextdit_action_dim=nextdit_cfg.get('action_dim', 3),
        nextdit_num_inference_steps=nextdit_cfg.get('num_inference_steps', 10),
        nextdit_guidance_scale=nextdit_cfg.get('guidance_scale', 1.0),
        nextdit_num_sample_trajs=nextdit_cfg.get('num_sample_trajs', 32),
        nextdit_dav2_ckpt_path=nextdit_cfg.get('dav2_ckpt_path', ''),
        nextdit_enable_gradient_checkpointing=nextdit_cfg.get('enable_gradient_checkpointing', True),

        verbose=verbose,
    )

    model = VLNPipeline(config)

    internnav_s1 = nextdit_cfg.get('internnav_system1_path', '') or nextdit_cfg.get('internnav_model_path', '')
    s1_ckpt = nextdit_cfg.get('pretrained_system1_path', '')
    if s1_ckpt and not internnav_s1 and model.nextdit_action_head is not None:
        s1_path = Path(s1_ckpt)
        if s1_path.exists():
            model.nextdit_action_head.load_pretrained_system1(
                str(s1_path),
                latent_queries=model.latent_queries,
            )
        else:
            logger.warning("System 1 pretrained weights not found: %s", s1_path)

    packing_enabled = llm_cfg.get('enable_packing', False)
    backbone_type = llm_cfg.get('backbone_type', 'qwen2_5_vl')
    if verbose:
        logger.info("VLN Pipeline built")
        logger.info("   Backbone -> %s (type=%s)", llm_cfg.get('model_path', './models/internnav_backbone'), backbone_type)
        logger.info("   SequencePacking -> enabled=%s", packing_enabled)
        logger.info(
            "   HeatmapVLN → enabled=%s, c_vit=%s, c_llm=%s, c_fused=%s, vit_layers=%s, llm_layers=%s",
            heatmap_cfg.get('enable', True),
            heatmap_cfg.get('c_vit', 1280),
            heatmap_cfg.get('c_llm', 3584),
            heatmap_cfg.get('c_fused', 256),
            heatmap_cfg.get('vit_layer_indices', [7, 15, 23, 31]),
            heatmap_cfg.get('llm_layer_indices', [6, 13, 20]),
        )
        logger.info(
            "   NextDiT ActionHead → enabled=%s",
            effective_action_head and nextdit_cfg.get('enabled', False),
        )
        if s1_ckpt:
            logger.info("   System1 pretrained → %s", s1_ckpt)

    return model


# ---------------------------------------------------------------------------
# Freeze / unfreeze utilities
# ---------------------------------------------------------------------------

def freeze_module(module: nn.Module, freeze: bool = True):
    for param in module.parameters():
        param.requires_grad = not freeze


_NEXTDIT_SUBMODULES = {
    'cond_projector': 'cond_projector',
    'traj_dit': 'traj_dit',
    'memory_encoder': 'memory_encoder',
    'rgb_resampler': 'rgb_resampler',
    'action_encoder': 'action_encoder',
    'action_decoder': 'action_decoder',
}

_BRIDGE_ONLY_MODULES = {'latent_queries', 'cond_projector'}


def _trainable_summary(model: VLNPipeline) -> dict[str, int]:
    summary: dict[str, int] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name == 'latent_queries':
            group = 'latent_queries'
        elif name.startswith('nextdit_action_head.'):
            parts = name.split('.')
            group = '.'.join(parts[:2]) if len(parts) > 1 else 'nextdit_action_head'
        elif name.startswith('heatmap_vln.'):
            parts = name.split('.')
            group = '.'.join(parts[:2]) if len(parts) > 1 else 'heatmap_vln'
        elif 'lora_' in name:
            group = 'vlm_lora'
        else:
            group = name.split('.', 1)[0]
        summary[group] = summary.get(group, 0) + param.numel()
    return summary


def _is_allowed_trainable_name(name: str, trainable_modules: set[str]) -> bool:
    if 'latent_queries' in trainable_modules and name == 'latent_queries':
        return True
    if 'nextdit_action_head' in trainable_modules and name.startswith('nextdit_action_head.'):
        return True
    for cfg_name, attr_name in _NEXTDIT_SUBMODULES.items():
        if cfg_name in trainable_modules and name.startswith(f'nextdit_action_head.{attr_name}.'):
            return True
    if 'heatmap_vln' in trainable_modules and name.startswith('heatmap_vln.'):
        return True
    if 'llm_projector' in trainable_modules and name.startswith('llm_projector.'):
        return True
    return ('lora' in trainable_modules or 'vlm_lora' in trainable_modules) and 'lora_' in name


def _assert_trainable_scope(model: VLNPipeline, stage_cfg: dict, logger) -> None:
    trainable = set(stage_cfg.get('trainable_modules', []))

    if stage_cfg.get('bridge_only', False):
        extra = sorted(trainable - _BRIDGE_ONLY_MODULES)
        missing = sorted(_BRIDGE_ONLY_MODULES - trainable)
        if extra or missing:
            raise ValueError(
                "bridge_only stages must train exactly latent_queries + cond_projector; "
                f"extra={extra}, missing={missing}"
            )

    if not stage_cfg.get('strict_trainable_modules', False):
        return

    violations = [
        name for name, param in model.named_parameters()
        if param.requires_grad and not _is_allowed_trainable_name(name, trainable)
    ]
    if violations:
        examples = ', '.join(violations[:8])
        raise RuntimeError(
            "Trainable parameter scope does not match trainable_modules. "
            f"Found {len(violations)} unexpected trainable tensors; examples: {examples}"
        )
    logger.info("  ✓ Strict trainable scope check passed")


def set_trainable_modules(model: VLNPipeline, stage_cfg: dict, logger):
    """Set trainable modules according to stage config."""
    freeze_module(model, freeze=True)

    trainable = stage_cfg.get('trainable_modules', [])

    if 'heatmap_vln' in trainable and hasattr(model, 'heatmap_vln') and model.heatmap_vln is not None:
        freeze_module(model.heatmap_vln.vit_dpt_fusion, freeze=False)
        freeze_module(model.heatmap_vln.llm_dpt_fusion, freeze=False)
        freeze_module(model.heatmap_vln.coarse, freeze=False)
        freeze_module(model.heatmap_vln.fine, freeze=False)
        logger.info("  ✓ Unfrozen: heatmap_vln (vit_dpt + llm_dpt + coarse + fine)")

    if 'nextdit_action_head' in trainable:
        if hasattr(model, 'nextdit_action_head') and model.nextdit_action_head is not None:
            freeze_module(model.nextdit_action_head, freeze=False)
            if hasattr(model.nextdit_action_head, 'rgb_model'):
                model.nextdit_action_head.rgb_model.requires_grad_(False)
            logger.info("  ✓ Unfrozen: nextdit_action_head (rgb_model kept frozen)")

    if 'latent_queries' in trainable and hasattr(model, 'latent_queries') and model.latent_queries is not None:
        model.latent_queries.requires_grad_(True)
        logger.info("  ✓ Unfrozen: latent_queries")

    # Fine-grained NextDiT sub-module unfreezing.
    if hasattr(model, 'nextdit_action_head') and model.nextdit_action_head is not None:
        for cfg_name, attr_name in _NEXTDIT_SUBMODULES.items():
            if cfg_name in trainable:
                submod = getattr(model.nextdit_action_head, attr_name, None)
                if submod is not None:
                    freeze_module(submod, freeze=False)
                    logger.info("  ✓ Unfrozen: nextdit_action_head.%s", attr_name)

    if 'llm_projector' in trainable and hasattr(model, 'llm_projector'):
        freeze_module(model.llm_projector, freeze=False)
        logger.info("  ✓ Unfrozen: llm_projector")

    vlm_backbone = getattr(model, 'vlm_backbone', getattr(model, 'qwen2_5_vl', None))
    if vlm_backbone is not None:
        freeze_module(vlm_backbone, freeze=True)
        if 'lora' in trainable or 'vlm_lora' in trainable:
            lora_count = 0
            for name, param in vlm_backbone.named_parameters():
                if 'lora_' in name:
                    param.requires_grad = True
                    lora_count += 1
            if lora_count > 0:
                logger.info(f"  ✓ Unfrozen: VLM LoRA ({lora_count} parameter tensors)")
            else:
                logger.warning("  ⚠️ LoRA in trainable_modules but no LoRA params found (model loaded?)")

    summary = _trainable_summary(model)
    if summary:
        logger.info("  Trainable parameter groups:")
        for group, count in sorted(summary.items()):
            logger.info("    - %s: %s params", group, f"{count:,}")
    else:
        logger.warning("  ⚠️ No trainable parameters after applying stage config")
    _assert_trainable_scope(model, stage_cfg, logger)


def apply_nextdit_warmup_freeze(model: VLNPipeline, cfg: dict, logger) -> int:
    """Apply warmup freeze: only cond_projector + latent_queries trainable.

    During warmup the bridge layers adapt to the LoRA-modified VLM
    representations before the downstream trajectory generator is
    unfrozen.  Must be called AFTER build_optimizer so all params are
    registered.  Returns warmup_steps (0 if disabled).
    """
    nextdit_cfg = cfg.get('model', {}).get('action_head', {}).get('nextdit', {})
    warmup_steps = nextdit_cfg.get('warmup_steps', 0)
    if warmup_steps <= 0:
        return 0
    nah = getattr(model, 'nextdit_action_head', None)
    if nah is None:
        return 0

    freeze_module(nah, freeze=True)
    freeze_module(nah.cond_projector, freeze=False)

    cp_params = sum(p.numel() for p in nah.cond_projector.parameters() if p.requires_grad)
    lq_params = model.latent_queries.numel() if (
        hasattr(model, 'latent_queries') and model.latent_queries is not None
        and model.latent_queries.requires_grad
    ) else 0

    logger.info(
        "🔥 NextDiT warmup active: first %d steps only train cond_projector (%s params) "
        "+ latent_queries (%s params), rest of System 1 frozen",
        warmup_steps, f"{cp_params:,}", f"{lq_params:,}",
    )
    return warmup_steps


def end_nextdit_warmup(model: VLNPipeline, logger, stage_cfg: dict | None = None):
    """Unfreeze System 1 modules according to trainable_modules after warmup.

    Unlike the old behaviour that blindly unfroze everything, this now
    respects *stage_cfg['trainable_modules']* so that modules listed in
    *frozen_modules* (e.g. memory_encoder, rgb_resampler) stay frozen.
    Falls back to unfreezing the whole head when no stage_cfg is given.
    """
    nah = getattr(model, 'nextdit_action_head', None)
    if nah is None:
        return

    trainable = stage_cfg.get('trainable_modules', []) if stage_cfg else []

    if 'nextdit_action_head' in trainable or not trainable:
        # Legacy / full unfreeze
        freeze_module(nah, freeze=False)
        if hasattr(nah, 'rgb_model'):
            nah.rgb_model.requires_grad_(False)
    else:
        # Selective unfreeze: only named sub-modules
        _submod_map = {
            'cond_projector': 'cond_projector',
            'traj_dit': 'traj_dit',
            'memory_encoder': 'memory_encoder',
            'rgb_resampler': 'rgb_resampler',
            'action_encoder': 'action_encoder',
            'action_decoder': 'action_decoder',
        }
        for cfg_name, attr_name in _submod_map.items():
            if cfg_name in trainable:
                submod = getattr(nah, attr_name, None)
                if submod is not None:
                    freeze_module(submod, freeze=False)

    total = sum(p.numel() for p in nah.parameters() if p.requires_grad)
    logger.info(
        "🔓 NextDiT warmup complete: unfrozen System 1 modules (%s trainable params)",
        f"{total:,}",
    )

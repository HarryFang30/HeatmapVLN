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

_INTERNNAV_SYSTEM1_REQUIRED_PREFIXES = (
    'cond_projector.',
    'traj_dit.',
    'memory_encoder.',
    'rgb_model.',
    'rgb_resampler.',
    'action_encoder.',
    'action_decoder.',
)


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
    pano_adapter_cfg = nextdit_cfg.get('pano_latent_adapter', {})
    past_plan_action_cfg = nextdit_cfg.get('past_plan_action', {})
    stop_head_cfg = model_cfg.get('stop_head', {})
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
        llm_frozen_traj_inference_mode=llm_cfg.get('frozen_traj_inference_mode', False),
        llm_traj_last_hidden_state_only=llm_cfg.get('traj_last_hidden_state_only', False),

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
        heatmap_decoder_mode=heatmap_cfg.get('decoder_mode', 'legacy'),
        heatmap_pose_free_config=heatmap_cfg.get('pose_free', None),
        heatmap_restore_vit_spatial_layout=heatmap_cfg.get(
            'restore_vit_spatial_layout',
            False,
        ),
        heatmap_coarse_logit_residual=heatmap_cfg.get(
            'coarse_logit_residual',
            False,
        ),
        heatmap_joint_panorama_inference=heatmap_cfg.get(
            'joint_panorama_inference',
            False,
        ),

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

        pano_latent_adapter_enabled=pano_adapter_cfg.get('enabled', False),
        pano_latent_adapter_hidden_dim=pano_adapter_cfg.get('hidden_dim', 1024),
        pano_latent_adapter_dropout=pano_adapter_cfg.get('dropout', 0.0),
        pano_latent_adapter_checkpoint_path=pano_adapter_cfg.get('pretrained_path', ''),
        pano_latent_adapter_strict_load=pano_adapter_cfg.get('strict_load', True),

        past_plan_action_enabled=past_plan_action_cfg.get('enabled', False),
        past_plan_action_plan_dim=past_plan_action_cfg.get('plan_dim', 768),
        past_plan_action_memory_dim=past_plan_action_cfg.get('memory_dim', 256),
        past_plan_action_bridge_heads=past_plan_action_cfg.get('bridge_heads', 8),

        stop_head_enabled=stop_head_cfg.get('enabled', False),
        stop_head_hidden_dim=stop_head_cfg.get('hidden_dim', 512),
        stop_head_dropout=stop_head_cfg.get('dropout', 0.1),
        stop_head_focal_gamma=stop_head_cfg.get('focal_gamma', 2.0),
        stop_head_focal_alpha=stop_head_cfg.get('focal_alpha', 0.5),
        stop_head_pos_weight=stop_head_cfg.get('pos_weight', 1.0),
        stop_head_bce_mix=stop_head_cfg.get('bce_mix', 0.5),

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
        if pano_adapter_cfg.get('enabled', False):
            logger.info(
                "   Pano latent adapter → enabled=True, hidden_dim=%s, pretrained=%s",
                pano_adapter_cfg.get('hidden_dim', 1024),
                pano_adapter_cfg.get('pretrained_path', '') or '<none>',
            )
        if s1_ckpt:
            logger.info("   System1 pretrained → %s", s1_ckpt)

    return model


def assert_complete_internnav_system1_load(
    model: VLNPipeline,
    *,
    logger: logging.Logger | None = None,
) -> int:
    """Refuse to use a partially initialized InternNav System1.

    ``VLNPipeline`` records every shape-compatible tensor copied from the
    released InternNav checkpoint.  Compare that audit against the complete
    local System1 state before any of those modules are frozen for training.
    """
    target_logger = logger or logging.getLogger(__name__)
    head = getattr(model, 'nextdit_action_head', None)
    audit = getattr(model, '_internnav_system1_load_audit', None)
    if head is None:
        raise RuntimeError('Model has no NextDiT action head')
    if not isinstance(audit, dict):
        raise RuntimeError(
            'InternNav System1 weights were not loaded. Configure '
            'model.action_head.nextdit.internnav_model_path with the released '
            'InternNav model directory.'
        )

    head_keys = set(head.state_dict())
    missing_modules = [
        prefix.removesuffix('.')
        for prefix in _INTERNNAV_SYSTEM1_REQUIRED_PREFIXES
        if not any(key.startswith(prefix) for key in head_keys)
    ]
    required_keys = {
        key
        for key in head_keys
        if key.startswith(_INTERNNAV_SYSTEM1_REQUIRED_PREFIXES)
    }
    loaded_keys = set(audit.get('loaded_keys') or ())
    missing_keys = sorted(required_keys - loaded_keys)
    latent_queries_loaded = bool(audit.get('latent_queries_loaded', False))

    if missing_modules or not required_keys or not latent_queries_loaded or missing_keys:
        missing_preview = ', '.join(missing_keys[:10])
        raise RuntimeError(
            'InternNav System1 load is incomplete; refusing to freeze random or '
            'partially initialized weights. '
            f"source={audit.get('source', '<unknown>')} "
            f'latent_queries_loaded={latent_queries_loaded} '
            f'missing_modules={missing_modules} missing_required={len(missing_keys)}'
            + (f' first_missing=[{missing_preview}]' if missing_preview else '')
        )

    target_logger.info(
        'Verified complete frozen InternNav System1 load from %s: %d required tensors + latent_queries',
        audit.get('source', '<unknown>'),
        len(required_keys),
    )
    return len(required_keys)


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


def _matches_exact_or_dotted_prefix(name: str, prefix: str) -> bool:
    """Match a leaf exactly or descendants of a module prefix only."""

    return name == prefix or (prefix.endswith('.') and name.startswith(prefix))


def _trainable_summary(model: VLNPipeline) -> dict[str, int]:
    summary: dict[str, int] = {}
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if name == 'latent_queries':
            group = 'latent_queries'
        elif name.startswith('past_plan_action.'):
            parts = name.split('.')
            group = '.'.join(parts[:2])
        elif name.startswith('pano_latent_adapter.'):
            group = 'pano_latent_adapter'
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
    if 'pano_latent_adapter' in trainable_modules and name.startswith('pano_latent_adapter.'):
        return True
    if 'stop_head' in trainable_modules and name.startswith('stop_head.'):
        return True
    if (
        'future_heatmap_head' in trainable_modules
        and name.startswith('past_plan_action.future_head.')
    ):
        return True
    if (
        'past_plan_bridge' in trainable_modules
        and name.startswith('past_plan_action.bridge.')
    ):
        return True
    if 'heatmap_memory_and_decoder' in trainable_modules:
        return any(
            _matches_exact_or_dotted_prefix(name, prefix)
            for prefix in (
                'heatmap_vln.coarse.proj_history.',
                'heatmap_vln.coarse.proj_traj.',
                'heatmap_vln.coarse.pos_embed',
                'heatmap_vln.coarse.self_attn.',
                'heatmap_vln.coarse.heatmap_head.',
                'heatmap_vln.coarse.vis_head.',
                'heatmap_vln.fine.',
            )
        )
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

    ppa_stage = stage_cfg.get('past_plan_action_stage')
    if ppa_stage is not None:
        chain = getattr(model, 'past_plan_action', None)
        past_head = getattr(model, 'heatmap_vln', None)
        action_head = getattr(model, 'nextdit_action_head', None)
        if chain is None or past_head is None or action_head is None:
            raise RuntimeError(
                "Past→Plan→Action stage requires model.past_plan_action, "
                "heatmap_vln, and nextdit_action_head"
            )
        from src.models.past_plan_action_training import (
            configure_past_plan_action_stage,
        )

        audit = configure_past_plan_action_stage(
            stage=ppa_stage,
            chain=chain,
            past_head=past_head,
            native_action_head=action_head,
            native_cond_projector=action_head.cond_projector,
            other_frozen_modules=tuple(
                module
                for module in (
                    getattr(model, 'vlm_backbone', None),
                    getattr(model, 'llm_projector', None),
                )
                if module is not None
            ),
        )
        logger.info(
            "  ✓ Past→Plan→Action %s scope: %d tensors / %s parameters",
            ppa_stage,
            audit.trainable_tensors,
            f"{audit.trainable_parameters:,}",
        )
        summary = _trainable_summary(model)
        for group, count in sorted(summary.items()):
            logger.info("    - %s: %s params", group, f"{count:,}")
        _assert_trainable_scope(model, stage_cfg, logger)
        return

    if 'heatmap_vln' in trainable and hasattr(model, 'heatmap_vln') and model.heatmap_vln is not None:
        if model.heatmap_vln.pose_free_matcher is not None:
            freeze_module(model.heatmap_vln.pose_free_matcher, freeze=False)
            logger.info("  ✓ Unfrozen: heatmap_vln (pose_free_matcher)")
        else:
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

    if (
        'pano_latent_adapter' in trainable
        and hasattr(model, 'pano_latent_adapter')
        and model.pano_latent_adapter is not None
    ):
        freeze_module(model.pano_latent_adapter, freeze=False)
        logger.info("  ✓ Unfrozen: pano_latent_adapter")

    if 'stop_head' in trainable:
        if getattr(model, 'stop_head', None) is None:
            raise RuntimeError("stop_head is trainable but model.stop_head is disabled")
        freeze_module(model.stop_head, freeze=False)
        logger.info("  ✓ Unfrozen: System2 STOP head")

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

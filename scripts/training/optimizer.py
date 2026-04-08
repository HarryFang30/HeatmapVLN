"""
Optimizer, learning-rate scheduler, and temperature schedule construction.
"""

import math
from typing import Dict

import torch
import torch.nn as nn

from src.models.pipeline import VLNPipeline


def build_optimizer(model: VLNPipeline, cfg: Dict, stage_cfg: Dict) -> torch.optim.Optimizer:
    """Build a per-module optimizer with layered learning rates and weight decay."""
    optim_cfg = cfg['optim']
    param_groups = []

    default_wd = optim_cfg.get('weight_decay', 1e-2)
    projector_wd = optim_cfg.get('projector_weight_decay', default_wd)

    def get_param_groups_with_wd(module, lr, name, wd):
        decay_params = []
        no_decay_params = []
        for n, p in module.named_parameters():
            if not p.requires_grad:
                continue
            if 'bias' in n or 'norm' in n.lower() or 'ln' in n.lower() or n == 'layer_weights':
                no_decay_params.append(p)
            else:
                decay_params.append(p)

        groups = []
        if decay_params:
            groups.append({
                'params': decay_params,
                'lr': lr,
                'weight_decay': wd,
                'name': f'{name}_decay'
            })
        if no_decay_params:
            groups.append({
                'params': no_decay_params,
                'lr': lr,
                'weight_decay': 0.0,
                'name': f'{name}_no_decay'
            })
        return groups

    # HeatmapVLN v2 param groups
    heatmap_lr = optim_cfg.get('heatmap_lr', 2e-4)
    vis_head_lr = optim_cfg.get('vis_head_lr', heatmap_lr)
    if hasattr(model, 'heatmap_vln') and model.heatmap_vln is not None:
        for name, submodule in [
            ('vit_dpt_fusion', model.heatmap_vln.vit_dpt_fusion),
            ('llm_dpt_fusion', model.heatmap_vln.llm_dpt_fusion),
            ('fine',           model.heatmap_vln.fine),
        ]:
            groups = get_param_groups_with_wd(submodule, heatmap_lr, f'heatmap_{name}', default_wd)
            if groups:
                param_groups.extend(groups)
                print(f"  Param group: heatmap_{name} (lr={heatmap_lr}, wd={default_wd})")

        coarse_module = model.heatmap_vln.coarse
        vis_head_params_decay = []
        vis_head_params_no_decay = []
        coarse_rest_decay = []
        coarse_rest_no_decay = []
        for n, p in coarse_module.named_parameters():
            if not p.requires_grad:
                continue
            is_vis_head = n.startswith('vis_head.')
            is_no_decay = 'bias' in n or 'norm' in n.lower() or 'ln' in n.lower()
            if is_vis_head:
                (vis_head_params_no_decay if is_no_decay else vis_head_params_decay).append(p)
            else:
                (coarse_rest_no_decay if is_no_decay else coarse_rest_decay).append(p)
        if coarse_rest_decay:
            param_groups.append({'params': coarse_rest_decay, 'lr': heatmap_lr, 'weight_decay': default_wd, 'name': 'heatmap_coarse_decay'})
        if coarse_rest_no_decay:
            param_groups.append({'params': coarse_rest_no_decay, 'lr': heatmap_lr, 'weight_decay': 0.0, 'name': 'heatmap_coarse_no_decay'})
        if vis_head_params_decay:
            param_groups.append({'params': vis_head_params_decay, 'lr': vis_head_lr, 'weight_decay': default_wd, 'name': 'heatmap_vis_head_decay'})
        if vis_head_params_no_decay:
            param_groups.append({'params': vis_head_params_no_decay, 'lr': vis_head_lr, 'weight_decay': 0.0, 'name': 'heatmap_vis_head_no_decay'})
        n_vis = len(vis_head_params_decay) + len(vis_head_params_no_decay)
        n_coarse = len(coarse_rest_decay) + len(coarse_rest_no_decay)
        print(f"  Param group: heatmap_coarse (lr={heatmap_lr}, {n_coarse} params)")
        print(f"  Param group: heatmap_vis_head (lr={vis_head_lr}, {n_vis} params)")

    # NextDiT Action Head — split submodules for per-component learning rates
    action_lr = optim_cfg.get('action_lr', 1e-4)
    nextdit_lr = optim_cfg.get('nextdit_action_lr', action_lr)
    nextdit_cond_lr = optim_cfg.get('nextdit_cond_projector_lr', nextdit_lr * 3)
    memory_encoder_lr = optim_cfg.get('memory_encoder_lr', nextdit_lr)
    rgb_resampler_lr = optim_cfg.get('rgb_resampler_lr', nextdit_lr)
    if hasattr(model, 'nextdit_action_head') and model.nextdit_action_head is not None:
        nah = model.nextdit_action_head
        dedicated_param_ids = set()

        cp_groups = get_param_groups_with_wd(nah.cond_projector, nextdit_cond_lr, 'nextdit_cond_projector', default_wd)
        if cp_groups:
            param_groups.extend(cp_groups)
            print(f"  Param group: nextdit_cond_projector (lr={nextdit_cond_lr}, wd={default_wd})")
        dedicated_param_ids.update(id(p) for p in nah.cond_projector.parameters())

        me_groups = get_param_groups_with_wd(nah.memory_encoder, memory_encoder_lr, 'nextdit_memory_encoder', default_wd)
        if me_groups:
            param_groups.extend(me_groups)
            print(f"  Param group: nextdit_memory_encoder (lr={memory_encoder_lr}, wd={default_wd})")
        dedicated_param_ids.update(id(p) for p in nah.memory_encoder.parameters())

        rr_groups = get_param_groups_with_wd(nah.rgb_resampler, rgb_resampler_lr, 'nextdit_rgb_resampler', default_wd)
        if rr_groups:
            param_groups.extend(rr_groups)
            print(f"  Param group: nextdit_rgb_resampler (lr={rgb_resampler_lr}, wd={default_wd})")
        dedicated_param_ids.update(id(p) for p in nah.rgb_resampler.parameters())

        rest_decay, rest_no_decay = [], []
        for n, p in nah.named_parameters():
            if not p.requires_grad or id(p) in dedicated_param_ids:
                continue
            if 'bias' in n or 'norm' in n.lower() or 'ln' in n.lower():
                rest_no_decay.append(p)
            else:
                rest_decay.append(p)
        if rest_decay:
            param_groups.append({'params': rest_decay, 'lr': nextdit_lr, 'weight_decay': default_wd, 'name': 'nextdit_rest_decay'})
        if rest_no_decay:
            param_groups.append({'params': rest_no_decay, 'lr': nextdit_lr, 'weight_decay': 0.0, 'name': 'nextdit_rest_no_decay'})
        if rest_decay or rest_no_decay:
            print(f"  Param group: nextdit_rest (lr={nextdit_lr}, wd={default_wd})")

    # Latent Queries
    if hasattr(model, 'latent_queries') and model.latent_queries is not None:
        latent_q_lr = optim_cfg.get('latent_queries_lr', action_lr)
        param_groups.append({
            'params': [model.latent_queries],
            'lr': latent_q_lr,
            'weight_decay': 0.0,
            'name': 'latent_queries',
        })
        print(f"  Param group: latent_queries (lr={latent_q_lr}, wd=0)")

    # LLM Projector
    proj_lr = optim_cfg.get('llm_projector_lr', 3e-5)
    if hasattr(model, 'llm_projector'):
        groups = get_param_groups_with_wd(model.llm_projector, proj_lr, 'llm_projector', projector_wd)
        if groups:
            param_groups.extend(groups)
            print(f"  Param group: llm_projector (lr={proj_lr}, wd={projector_wd})")

    # VLM backbone LoRA parameters
    lora_lr = optim_cfg.get('lora_lr', 1e-5)
    vlm_backbone = getattr(model, 'vlm_backbone', getattr(model, 'qwen2_5_vl', None))
    if vlm_backbone is not None:
        lora_params = [p for n, p in vlm_backbone.named_parameters()
                       if p.requires_grad and 'lora_' in n]
        if lora_params:
            param_groups.append({
                'params': lora_params,
                'lr': lora_lr,
                'weight_decay': 0.0,
                'name': 'vlm_lora'
            })
            print(f"  Param group: vlm_lora (lr={lora_lr}, wd=0.0, params={len(lora_params)})")

    if not param_groups:
        raise ValueError("No trainable parameters found!")

    optimizer = torch.optim.AdamW(param_groups)
    return optimizer


def build_scheduler(optimizer, cfg: Dict, total_steps: int):
    """Build warmup + cosine annealing LR scheduler."""
    optim_cfg = cfg['optim']
    warmup_steps = int(total_steps * optim_cfg['warmup_ratio'])

    from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

    warmup = LinearLR(
        optimizer,
        start_factor=0.1,
        total_iters=warmup_steps
    )

    cosine = CosineAnnealingLR(
        optimizer,
        T_max=max(1, total_steps - warmup_steps),
        eta_min=optim_cfg.get('min_lr', 1e-6)
    )

    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup, cosine],
        milestones=[warmup_steps]
    )

    return scheduler


def get_heatmap_temperature(cfg: Dict, step: int, total_steps: int) -> float:
    """Return current soft-argmax temperature based on optimization step."""
    heatmap_loss_cfg = cfg.get('loss', {}).get('heatmap_vln', {})
    base_temperature = float(heatmap_loss_cfg.get('temperature', 1.0))
    schedule_cfg = heatmap_loss_cfg.get('temperature_schedule', {})

    if not schedule_cfg or not schedule_cfg.get('enabled', False):
        return base_temperature

    start_temp = float(schedule_cfg.get('start', base_temperature))
    end_temp = float(schedule_cfg.get('end', start_temp))
    mode = str(schedule_cfg.get('mode', 'cosine')).lower()

    if total_steps <= 1:
        return end_temp

    progress = min(max(step / max(total_steps - 1, 1), 0.0), 1.0)

    if mode == 'linear':
        interp = progress
    else:
        interp = 0.5 * (1.0 - math.cos(math.pi * progress))

    return start_temp + (end_temp - start_temp) * interp

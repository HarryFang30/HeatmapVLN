"""
Action Module — NextDiT System 1 (InternNav-compatible)

Components:
- NextDiTActionHead: Flow matching trajectory prediction via NextDiT + cross-attention
- NextDiTActionConfig: Configuration dataclass for NextDiTActionHead

Usage:
    from src.models.action import NextDiTActionHead, NextDiTActionConfig

    cfg = NextDiTActionConfig(vlm_hidden_dim=3584, latent_emb_size=768)
    action_head = NextDiTActionHead(cfg)
    trajectory = action_head.get_trajectory(traj_hidden_states)  # (B, T, 3)
"""

from .nextdit_action_head import NextDiTActionHead, NextDiTActionConfig

__all__ = [
    'NextDiTActionHead',
    'NextDiTActionConfig',
]

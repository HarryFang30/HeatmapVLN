"""
NextDiT Action Head — DualVLN System 1 for HeatmapVLN
======================================================

Ported from InternNav's InternVLA-N1 (nextdit_async mode).

Architecture (async):
    traj_hidden_states (B, n_query, vlm_hidden_dim)
        -> cond_projector -> (B, n_query, latent_emb_size)
    traj_images [pixel_goal, current] (B, 2, H, W, 3)
        -> DINOv2 (frozen) -> (B, 2, 256, 384)
        -> flatten -> (B, 512, 384)
        -> MemoryEncoder (self-attn) -> (B, 512, 384)
        -> cat(original, encoded) -> (B, 512, 768)
        -> QFormer (32 queries) -> memory_tokens (B, 32, 768)
    latents = cat([memory_tokens, traj_cond]) -> (B, 36, latent_emb_size)
    NextDiT + FlowMatch denoising loop -> trajectory (B, T, 3)

Training loss: Flow Matching velocity prediction MSE.
Inference: Euler ODE solver with Classifier-Free Guidance.
"""

import logging
from dataclasses import dataclass
from typing import Dict, Optional, List

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor

from .nextdit import NextDiTCrossAttn, NextDiTCrossAttnConfig
from .nextdit import SinusoidalPositionalEncoding, MemoryEncoder, QFormer

logger = logging.getLogger(__name__)

LATENT_EMB_SIZE = 768
RESNET_MEAN = [0.485, 0.456, 0.406]
RESNET_STD = [0.229, 0.224, 0.225]


@dataclass
class NextDiTActionConfig:
    """Configuration for NextDiTActionHead."""
    vlm_hidden_dim: int = 4096
    latent_emb_size: int = LATENT_EMB_SIZE
    n_query: int = 4
    dit_dim: int = 384
    dit_layers: int = 12
    dit_heads: int = 6
    dit_kv_heads: int = 6
    predict_steps: int = 32
    action_dim: int = 3
    num_inference_steps: int = 10
    guidance_scale: float = 1.0
    num_sample_trajs: int = 32
    memory_encoder_hidden: int = 384
    memory_encoder_heads: int = 6
    memory_encoder_layers: int = 3
    qformer_num_query: int = 32
    qformer_hidden: int = LATENT_EMB_SIZE
    qformer_layers: int = 3
    qformer_heads: int = 12
    dav2_ckpt_path: str = ""
    enable_gradient_checkpointing: bool = True


class NextDiTActionHead(nn.Module):
    """
    DualVLN System 1 action head using NextDiT + Flow Matching + Visual Memory.

    Replaces the TransformerActionHead in HeatmapVLN pipeline.
    """

    def __init__(self, config: NextDiTActionConfig):
        super().__init__()
        self.config = config

        # ==================== Condition Projector ====================
        # Adapts Qwen3.5 hidden_dim (4096) -> latent_emb_size (768)
        self.cond_projector = nn.Sequential(
            nn.Linear(config.vlm_hidden_dim, config.latent_emb_size),
            nn.GELU(approximate="tanh"),
            nn.Linear(config.latent_emb_size, config.latent_emb_size),
        )

        # ==================== Visual Memory (async) ====================
        self.rgb_model = self._build_dav2_backbone(config.dav2_ckpt_path)
        self.memory_encoder = MemoryEncoder(
            hidden_size=config.memory_encoder_hidden,
            num_heads=config.memory_encoder_heads,
            num_layers=config.memory_encoder_layers,
            max_len=512,
        )
        self.rgb_resampler = QFormer(
            num_query=config.qformer_num_query,
            hidden_size=config.qformer_hidden,
            num_layers=config.qformer_layers,
            num_heads=config.qformer_heads,
        )

        # ==================== NextDiT ====================
        dit_config = NextDiTCrossAttnConfig(
            in_channels=config.dit_dim,
            dim=config.dit_dim,
            n_layers=config.dit_layers,
            n_heads=config.dit_heads,
            n_kv_heads=config.dit_kv_heads,
            latent_embedding_size=config.latent_emb_size,
            learn_sigma=False,
            _gradient_checkpointing=config.enable_gradient_checkpointing,
        )
        self.traj_dit = NextDiTCrossAttn(dit_config)
        self.noise_scheduler = FlowMatchEulerDiscreteScheduler()

        # ==================== Action Encoder / Decoder ====================
        self.action_encoder = nn.Linear(config.action_dim, config.dit_dim, bias=True)
        self.pos_encoding = SinusoidalPositionalEncoding(config.dit_dim)
        self.action_decoder = nn.Linear(config.dit_dim, config.action_dim, bias=True)

        # ==================== ResNet normalization buffers ====================
        self.register_buffer(
            "_resnet_mean",
            torch.FloatTensor(RESNET_MEAN).view(1, 1, 3, 1, 1),
            persistent=False,
        )
        self.register_buffer(
            "_resnet_std",
            torch.FloatTensor(RESNET_STD).view(1, 1, 3, 1, 1),
            persistent=False,
        )

        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        logger.info(
            "NextDiTActionHead initialized: dit_layers=%d, dit_dim=%d, predict_steps=%d, "
            "total_params=%s, trainable_params=%s",
            config.dit_layers, config.dit_dim, config.predict_steps,
            f"{total_params:,}", f"{trainable_params:,}",
        )

    @staticmethod
    def _build_dav2_backbone(ckpt_path: str) -> nn.Module:
        """Load DepthAnythingV2's DINOv2-vits pretrained encoder (frozen)."""
        from .nextdit.depth_anything.dpt import DepthAnythingV2

        model_configs = {
            "vits": {
                "encoder": "vits",
                "features": 64,
                "out_channels": [48, 96, 192, 384],
            }
        }
        dav2_model = DepthAnythingV2(**model_configs["vits"])
        if ckpt_path:
            state_dict = torch.load(ckpt_path, map_location="cpu")
            dav2_model.load_state_dict(state_dict)
            logger.info("Loaded DepthAnythingV2 weights from %s", ckpt_path)
        else:
            logger.warning("No DepthAnythingV2 checkpoint provided, using random init")

        rgb_model = dav2_model.pretrained
        rgb_model.requires_grad_(False)
        rgb_model.eval()
        return rgb_model

    # ==================== Visual Memory Forward ====================

    def _encode_visual_memory(self, traj_images: torch.Tensor) -> torch.Tensor:
        """
        Process [pixel_goal, current] images through visual memory pipeline.

        Args:
            traj_images: (B, 2, H, W, 3) — [pixel_goal_image, current_image], float [0,1]

        Returns:
            memory_tokens: (B, qformer_num_query, latent_emb_size)
        """
        B = traj_images.shape[0]
        images_dp = traj_images.permute(0, 1, 4, 2, 3)  # (B, 2, 3, H, W)
        images_dp_norm = (images_dp - self._resnet_mean) / self._resnet_std

        with torch.no_grad():
            images_dp_feat = (
                self.rgb_model
                .get_intermediate_layers(images_dp_norm.flatten(0, 1))[0]
                .unflatten(dim=0, sizes=(B, -1))
            )
        # images_dp_feat: (B, 2, num_patches, 384)

        flat_feat = images_dp_feat.flatten(1, 2)  # (B, 2*num_patches, 384)
        encoded_feat = self.memory_encoder(flat_feat)  # (B, 2*num_patches, 384)

        # Residual concatenation along feature dim
        memory_feat = torch.cat([flat_feat, encoded_feat], dim=-1)  # (B, 2*num_patches, 768)

        memory_tokens = self.rgb_resampler(memory_feat)  # (B, qformer_num_query, 768)
        return memory_tokens

    # ==================== Condition Fusion ====================

    def _fuse_conditions(
        self,
        traj_hidden_states: torch.Tensor,
        traj_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Fuse System 2 latent queries with visual memory.

        Args:
            traj_hidden_states: (B, n_query, vlm_hidden_dim) from Qwen3.5
            traj_images: (B, 2, H, W, 3) optional visual memory images

        Returns:
            latents: (B, N, latent_emb_size) — fused condition for NextDiT
        """
        traj_cond = self.cond_projector(traj_hidden_states)  # (B, n_query, 768)

        if traj_images is not None:
            memory_tokens = self._encode_visual_memory(traj_images)  # (B, 32, 768)
            latents = torch.cat([memory_tokens, traj_cond], dim=1)  # (B, 32+n_query, 768)
        else:
            latents = traj_cond

        return latents

    # ==================== Flow Matching Training ====================

    def _get_sigmas(self, timesteps: torch.Tensor, device, n_dim: int = 3, dtype=torch.float32):
        """Look up sigma values from the scheduler for given timesteps."""
        sigmas = self.noise_scheduler.sigmas.to(device=device, dtype=dtype)
        schedule_timesteps = self.noise_scheduler.timesteps.to(device=device)
        timesteps = timesteps.to(device)
        step_indices = [(schedule_timesteps == t).nonzero().item() for t in timesteps]
        sigma = sigmas[step_indices].flatten()
        while len(sigma.shape) < n_dim:
            sigma = sigma.unsqueeze(-1)
        return sigma

    def compute_loss(
        self,
        traj_hidden_states: torch.Tensor,
        gt_trajectory: torch.Tensor,
        traj_images: Optional[torch.Tensor] = None,
        trajectory_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute flow matching training loss.

        Args:
            traj_hidden_states: (B, n_query, vlm_hidden_dim)
            gt_trajectory: (B, predict_steps, action_dim) — relative poses
            traj_images: (B, 2, H, W, 3) — [pixel_goal, current]
            trajectory_valid: (B,) — mask for valid samples

        Returns:
            Dict with 'loss' key
        """
        latents = self._fuse_conditions(traj_hidden_states, traj_images)

        bsz = gt_trajectory.shape[0]
        device = gt_trajectory.device
        dtype = gt_trajectory.dtype

        # Sample noise
        noise = torch.randn_like(gt_trajectory)

        # Sample timesteps: u ~ U(0, 1)
        u = torch.rand(size=(bsz,), device="cpu")
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(device=device)
        sigmas = self._get_sigmas(timesteps, device, n_dim=gt_trajectory.ndim, dtype=dtype)

        # Flow matching interpolation: X_u = (1 - sigma) * X_0 + sigma * epsilon
        noisy_trajectory = (1 - sigmas) * gt_trajectory + sigmas * noise

        # Encode noisy trajectory
        action_features = self.action_encoder(noisy_trajectory)  # (B, T, dit_dim)
        pos_ids = torch.arange(gt_trajectory.shape[1], device=device).reshape(1, -1).repeat(bsz, 1)
        pos_embed = self.pos_encoding(pos_ids)
        action_features = action_features + pos_embed

        # NextDiT forward
        noise_pred = self.traj_dit(
            x=action_features,
            timestep=timesteps,
            z_latents=latents,
        )
        noise_pred = self.action_decoder(noise_pred)  # (B, T, action_dim)

        # Velocity target: v = epsilon - X_0
        target = noise - gt_trajectory
        loss = F.mse_loss(noise_pred.float(), target.float(), reduction="none")

        # Apply validity mask
        if trajectory_valid is not None:
            mask = trajectory_valid.float()
            if mask.sum() > 0:
                per_sample_loss = loss.mean(dim=(1, 2))
                loss_val = (per_sample_loss * mask).sum() / mask.sum()
            else:
                loss_val = loss.mean()
        else:
            loss_val = loss.mean()

        return {"loss": loss_val}

    # ==================== Flow Matching Inference ====================

    @torch.no_grad()
    def generate_traj(
        self,
        traj_hidden_states: torch.Tensor,
        traj_images: Optional[torch.Tensor] = None,
        predict_step_nums: int = 32,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 10,
        num_sample_trajs: int = 32,
    ) -> torch.Tensor:
        """
        Generate trajectory via iterative flow matching denoising with CFG.

        Args:
            traj_hidden_states: (B, n_query, vlm_hidden_dim)
            traj_images: (B, 2, H, W, 3) — [pixel_goal, current]
            predict_step_nums: number of trajectory steps to predict
            guidance_scale: classifier-free guidance scale
            num_inference_steps: number of denoising steps
            num_sample_trajs: number of parallel trajectory samples

        Returns:
            latents: (B * num_sample_trajs, predict_step_nums, action_dim)
        """
        latents_cond = self._fuse_conditions(traj_hidden_states, traj_images)

        device = traj_hidden_states.device
        dtype = traj_hidden_states.dtype
        batch_size = traj_hidden_states.shape[0]

        # Classifier-Free Guidance: [unconditional, conditional]
        hidden_states_null = torch.zeros_like(latents_cond)
        hidden_states_input = torch.cat([hidden_states_null, latents_cond], dim=0)
        hidden_states_input = hidden_states_input.repeat_interleave(num_sample_trajs, dim=0)

        # Initialize random noise
        traj_latents = randn_tensor(
            shape=(batch_size * num_sample_trajs, predict_step_nums, self.config.action_dim),
            generator=None,
            device=device,
            dtype=dtype,
        )

        # Set up scheduler
        scheduler = FlowMatchEulerDiscreteScheduler()
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)

        # Iterative denoising
        for t in scheduler.timesteps:
            latent_features = self.action_encoder(traj_latents)
            pos_ids = (
                torch.arange(latent_features.shape[1], device=device)
                .reshape(1, -1)
                .repeat(batch_size * num_sample_trajs, 1)
            )
            pos_embed = self.pos_encoding(pos_ids)
            latent_features = latent_features + pos_embed

            # Double for CFG
            latent_model_input = latent_features.repeat(2, 1, 1)
            if hasattr(scheduler, "scale_model_input"):
                latent_model_input = scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.traj_dit(
                x=latent_model_input,
                timestep=t.unsqueeze(0).expand(latent_model_input.shape[0]).to(device, torch.long),
                z_latents=hidden_states_input,
            )
            noise_pred = self.action_decoder(noise_pred)

            # CFG interpolation
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            traj_latents = scheduler.step(noise_pred, t, traj_latents).prev_sample

        return traj_latents

    @torch.no_grad()
    def get_trajectory(
        self,
        traj_hidden_states: torch.Tensor,
        traj_images: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        High-level inference interface matching TransformerActionHead API.

        Returns:
            trajectory: (B * num_sample_trajs, predict_steps, action_dim)
        """
        self.eval()
        return self.generate_traj(
            traj_hidden_states=traj_hidden_states,
            traj_images=traj_images,
            predict_step_nums=self.config.predict_steps,
            guidance_scale=self.config.guidance_scale,
            num_inference_steps=self.config.num_inference_steps,
            num_sample_trajs=self.config.num_sample_trajs,
        )

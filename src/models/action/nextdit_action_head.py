"""
NextDiT Action Head — DualVLN System 1 for HeatmapVLN
======================================================

Ported from InternNav's InternVLA-N1 (nextdit_async mode).

Architecture (async):
    traj_hidden_states (B, n_query, vlm_hidden_dim)
        -> cond_projector -> (B, n_query, latent_emb_size)
    traj_images [anchor, current] (B, 2, H, W, 3)
        or training sequence frames (B, N, H, W, 3)
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
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers import FlowMatchEulerDiscreteScheduler
from diffusers.utils.torch_utils import randn_tensor
from safetensors import safe_open

from .nextdit import MemoryEncoder, NextDiTCrossAttn, NextDiTCrossAttnConfig, QFormer, SinusoidalPositionalEncoding

logger = logging.getLogger(__name__)

LATENT_EMB_SIZE = 768
RESNET_MEAN = [0.485, 0.456, 0.406]
RESNET_STD = [0.229, 0.224, 0.225]


@dataclass
class NextDiTActionConfig:
    """Configuration for NextDiTActionHead."""

    vlm_hidden_dim: int = 3584
    latent_emb_size: int = LATENT_EMB_SIZE
    n_query: int = 4
    dit_dim: int = 384
    dit_layers: int = 12
    dit_heads: int = 6
    dit_kv_heads: int = 6
    dit_ffn_dim_multiplier: float | None = 2 / 3
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

    InternNav-compatible action head for HeatmapVLN pipeline.
    """

    def __init__(self, config: NextDiTActionConfig):
        super().__init__()
        self.config = config

        # ==================== Condition Projector ====================
        # Adapts backbone hidden_dim -> latent_emb_size (768)
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
            ffn_dim_multiplier=config.dit_ffn_dim_multiplier,
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
            config.dit_layers,
            config.dit_dim,
            config.predict_steps,
            f"{total_params:,}",
            f"{trainable_params:,}",
        )

    def load_pretrained_system1(
        self,
        ckpt_path: str,
        latent_queries: nn.Parameter | None = None,
    ) -> tuple[list[str], list[str], int]:
        """
        Load pretrained System 1 weights from extracted DualVLN checkpoint.

        The checkpoint keys use the naming convention of NextDiTActionHead's
        own sub-modules (e.g. ``traj_dit.*``, ``memory_encoder.*``), plus an
        optional ``latent_queries`` tensor that lives in the pipeline.

        Tensors whose shapes do not match (e.g. ``cond_projector.0.weight``
        when vlm_hidden_dim differs) are automatically skipped with a warning.

        Args:
            ckpt_path: path to ``.safetensors`` file.
            latent_queries: if provided, try to load ``latent_queries`` from
                the checkpoint into this parameter (lives in pipeline, not here).

        Returns:
            (missing_keys, skipped_keys, loaded_count)
        """
        ckpt_path = str(ckpt_path)
        logger.info("Loading System 1 pretrained weights from %s", ckpt_path)

        if ckpt_path.endswith(".safetensors"):
            with safe_open(ckpt_path, framework="pt", device="cpu") as f:
                ckpt_sd = {k: f.get_tensor(k) for k in f}
        else:
            ckpt_sd = torch.load(ckpt_path, map_location="cpu", weights_only=True)

        current_sd = self.state_dict()
        loaded_sd: dict[str, torch.Tensor] = {}
        skipped: list[str] = []

        for ckpt_key, ckpt_val in ckpt_sd.items():
            if ckpt_key == "latent_queries":
                if latent_queries is not None:
                    if latent_queries.shape == ckpt_val.shape:
                        latent_queries.data.copy_(ckpt_val)
                        logger.info("  Loaded latent_queries %s", tuple(ckpt_val.shape))
                    else:
                        skipped.append(
                            f"latent_queries: ckpt {tuple(ckpt_val.shape)} vs model {tuple(latent_queries.shape)}"
                        )
                continue

            if ckpt_key in current_sd:
                if current_sd[ckpt_key].shape == ckpt_val.shape:
                    loaded_sd[ckpt_key] = ckpt_val
                else:
                    skipped.append(
                        f"{ckpt_key}: ckpt {tuple(ckpt_val.shape)} vs model {tuple(current_sd[ckpt_key].shape)}"
                    )

        if skipped:
            logger.warning(
                "Skipped %d tensors due to shape mismatch:\n  %s",
                len(skipped),
                "\n  ".join(skipped),
            )

        missing, _unexpected = self.load_state_dict(loaded_sd, strict=False)
        real_missing = [k for k in missing if not k.startswith("_")]
        loaded_count = len(loaded_sd)

        logger.info(
            "System 1 weight loading complete: loaded=%d, skipped=%d, missing=%d",
            loaded_count,
            len(skipped),
            len(real_missing),
        )
        if real_missing:
            logger.info("  Missing keys (random init): %s", real_missing)

        return real_missing, skipped, loaded_count

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
            ckpt_file = Path(ckpt_path)
            if ckpt_file.is_file():
                state_dict = torch.load(str(ckpt_file), map_location="cpu", weights_only=True)
                dav2_model.load_state_dict(state_dict)
                logger.info("Loaded DepthAnythingV2 weights from %s", ckpt_file)
            else:
                logger.warning(
                    "DepthAnythingV2 checkpoint not found at %s; continuing without external DAV2 weights "
                    "and expecting System 1 checkpoint loading to overwrite them if available",
                    ckpt_path,
                )
        else:
            logger.info(
                "DepthAnythingV2 encoder created without dav2_ckpt_path; "
                "weights should be loaded from InternNav_Model safetensors (rgb_model.*) "
                "via pipeline internnav_model_path before evaluation."
            )

        rgb_model = dav2_model.pretrained
        rgb_model.requires_grad_(False)
        rgb_model.eval()
        return rgb_model

    # ==================== Visual Memory Forward ====================

    def _encode_visual_memory(self, traj_images: torch.Tensor) -> torch.Tensor:
        """
        Process [anchor, current] images through visual memory pipeline.

        Args:
            traj_images: (B, 2, H, W, 3) — [anchor_image, current_image], float [0,1]

        Returns:
            memory_tokens: (B, qformer_num_query, latent_emb_size)
        """
        B = traj_images.shape[0]
        images_dp = traj_images.permute(0, 1, 4, 2, 3)  # (B, 2, 3, H, W)
        model_dtype = next(self.rgb_model.parameters()).dtype
        images_dp_norm = (images_dp.to(dtype=model_dtype) - self._resnet_mean) / self._resnet_std

        with torch.no_grad():
            images_dp_feat = self.rgb_model.get_intermediate_layers(images_dp_norm.flatten(0, 1))[0].unflatten(
                dim=0, sizes=(B, -1)
            )
        # images_dp_feat: (B, 2, num_patches, 384)

        flat_feat = images_dp_feat.flatten(1, 2)  # (B, 2*num_patches, 384)
        encoded_feat = self.memory_encoder(flat_feat)  # (B, 2*num_patches, 384)

        # Residual concatenation along feature dim
        memory_feat = torch.cat([flat_feat, encoded_feat], dim=-1)  # (B, 2*num_patches, 768)

        memory_tokens = self.rgb_resampler(memory_feat)  # (B, qformer_num_query, 768)
        return memory_tokens

    @staticmethod
    def _expand_sequence_training_inputs(
        traj_hidden_states: torch.Tensor,
        gt_trajectory: torch.Tensor,
        traj_images: torch.Tensor | None,
        trajectory_valid: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None, torch.Tensor | None]:
        """Flatten InternNav-style multi-current training samples.

        Training may provide ``traj_images`` as ``(B, N, H, W, 3)`` and
        trajectories as ``(B, N, T, 3)``.  The first image is the fixed anchor;
        each of the N images becomes the current image for one trajectory loss.
        Evaluation still passes explicit pairs as ``(B, 2, H, W, 3)``.
        """
        if traj_images is None or traj_images.ndim != 5 or gt_trajectory.ndim != 4:
            return traj_hidden_states, gt_trajectory, traj_images, trajectory_valid

        batch_size, num_frames = traj_images.shape[:2]
        anchor_images = traj_images[:, 0:1].repeat(1, num_frames, 1, 1, 1).flatten(0, 1)
        current_images = traj_images.flatten(0, 1)
        traj_image_pairs = torch.stack([anchor_images, current_images], dim=1)

        traj_hidden_states = traj_hidden_states.unsqueeze(1).repeat(1, num_frames, 1, 1).flatten(0, 1)
        gt_trajectory = gt_trajectory.flatten(0, 1)
        if trajectory_valid is not None:
            trajectory_valid = trajectory_valid.flatten(0, 1)

        expected = batch_size * num_frames
        if traj_hidden_states.shape[0] != expected or gt_trajectory.shape[0] != expected:
            raise RuntimeError(
                "Failed to flatten sequence training inputs: "
                f"expected {expected}, got hidden={traj_hidden_states.shape[0]}, "
                f"trajectory={gt_trajectory.shape[0]}"
            )

        return traj_hidden_states, gt_trajectory, traj_image_pairs, trajectory_valid

    # ==================== Condition Fusion ====================

    def _fuse_conditions(
        self,
        traj_hidden_states: torch.Tensor,
        traj_images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Fuse System 2 latent queries with visual memory.

        Args:
            traj_hidden_states: (B, n_query, vlm_hidden_dim) from Qwen2.5-VL backbone
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

    def _fuse_projected_conditions(
        self,
        traj_cond: torch.Tensor,
        traj_images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Fuse already-projected 768-dim trajectory condition tokens.

        This is the direct adapter path: callers have already mapped System-2
        latents into ``config.latent_emb_size`` and intentionally skip
        ``cond_projector``.
        """
        if traj_cond.ndim != 3:
            raise ValueError(f"traj_cond must be [B,Q,D], got {tuple(traj_cond.shape)}")
        if traj_cond.shape[-1] != self.config.latent_emb_size:
            raise ValueError(f"Expected projected dim {self.config.latent_emb_size}, got {traj_cond.shape[-1]}")

        if traj_images is not None:
            memory_tokens = self._encode_visual_memory(traj_images)
            return torch.cat([memory_tokens.to(dtype=traj_cond.dtype), traj_cond], dim=1)
        return traj_cond

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

    def sample_flow_matching_inputs(
        self,
        gt_trajectory: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample the shared noisy trajectory, timestep, and velocity target."""
        bsz = gt_trajectory.shape[0]
        device = gt_trajectory.device
        dtype = gt_trajectory.dtype
        noise = torch.randn_like(gt_trajectory)
        u = torch.rand(size=(bsz,), device="cpu")
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long()
        timesteps = self.noise_scheduler.timesteps[indices].to(device=device)
        sigmas = self._get_sigmas(timesteps, device, n_dim=gt_trajectory.ndim, dtype=dtype)
        noisy_trajectory = (1 - sigmas) * gt_trajectory + sigmas * noise
        target_velocity = noise - gt_trajectory
        return noisy_trajectory, timesteps, target_velocity

    def predict_velocity_from_projected(
        self,
        traj_cond: torch.Tensor,
        noisy_trajectory: torch.Tensor,
        timesteps: torch.Tensor,
        traj_images: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Predict flow velocity from pre-projected NextDiT condition tokens."""
        latents = self._fuse_projected_conditions(traj_cond, traj_images)
        bsz = noisy_trajectory.shape[0]
        action_features = self.action_encoder(noisy_trajectory)
        pos_ids = torch.arange(noisy_trajectory.shape[1], device=noisy_trajectory.device).reshape(1, -1).repeat(bsz, 1)
        pos_embed = self.pos_encoding(pos_ids).to(dtype=action_features.dtype)
        action_features = action_features + pos_embed
        velocity = self.traj_dit(
            x=action_features,
            timestep=timesteps,
            z_latents=latents,
        )
        return self.action_decoder(velocity)

    @staticmethod
    def masked_velocity_mse(
        pred: torch.Tensor,
        target: torch.Tensor,
        trajectory_valid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        loss = F.mse_loss(pred.float(), target.float(), reduction="none")
        if trajectory_valid is None:
            return loss.mean()
        mask = trajectory_valid.float()
        if mask.sum() <= 0:
            return loss.sum() * 0.0
        per_sample_loss = loss.mean(dim=(1, 2))
        return (per_sample_loss * mask).sum() / mask.sum()

    def compute_loss_from_projected(
        self,
        traj_cond: torch.Tensor,
        gt_trajectory: torch.Tensor,
        traj_images: torch.Tensor | None = None,
        trajectory_valid: torch.Tensor | None = None,
        noisy_trajectory: torch.Tensor | None = None,
        timesteps: torch.Tensor | None = None,
        target_velocity: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """Compute flow-matching loss from already-projected conditions."""
        traj_cond, gt_trajectory, traj_images, trajectory_valid = self._expand_sequence_training_inputs(
            traj_cond,
            gt_trajectory,
            traj_images,
            trajectory_valid,
        )
        if noisy_trajectory is None or timesteps is None or target_velocity is None:
            noisy_trajectory, timesteps, target_velocity = self.sample_flow_matching_inputs(gt_trajectory)
        velocity = self.predict_velocity_from_projected(
            traj_cond,
            noisy_trajectory,
            timesteps,
            traj_images=traj_images,
        )
        return {
            "loss": self.masked_velocity_mse(
                velocity,
                target_velocity,
                trajectory_valid=trajectory_valid,
            )
        }

    def compute_loss(
        self,
        traj_hidden_states: torch.Tensor,
        gt_trajectory: torch.Tensor,
        traj_images: torch.Tensor | None = None,
        trajectory_valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        """
        Compute flow matching training loss.

        Args:
            traj_hidden_states: (B, n_query, vlm_hidden_dim)
            gt_trajectory: (B, predict_steps, action_dim) — relative poses
            traj_images: (B, 2, H, W, 3) — [anchor, current], or
                (B, N, H, W, 3) InternNav-style current-frame sequence.
            trajectory_valid: (B,) or (B, N) — mask for valid samples

        Returns:
            Dict with 'loss' key
        """
        traj_hidden_states, gt_trajectory, traj_images, trajectory_valid = self._expand_sequence_training_inputs(
            traj_hidden_states,
            gt_trajectory,
            traj_images,
            trajectory_valid,
        )
        latents = self._fuse_conditions(traj_hidden_states, traj_images)

        bsz = gt_trajectory.shape[0]
        device = gt_trajectory.device
        dtype = gt_trajectory.dtype

        # Sample noise
        noise = torch.randn_like(gt_trajectory)

        # Sample timesteps: u ~ U(0, 1).
        # IMPORTANT: use the same device as gt_trajectory so that torch.manual_seed
        # controls the GPU RNG state, making timestep sampling deterministic.
        u = torch.rand(size=(bsz,), device=device)
        indices = (u * self.noise_scheduler.config.num_train_timesteps).long().to(device=device)
        timesteps = self.noise_scheduler.timesteps.to(device=device)[indices]
        sigmas = self._get_sigmas(timesteps, device, n_dim=gt_trajectory.ndim, dtype=dtype)

        # Flow matching interpolation: X_u = (1 - sigma) * X_0 + sigma * epsilon
        noisy_trajectory = (1 - sigmas) * gt_trajectory + sigmas * noise

        # Encode noisy trajectory
        action_features = self.action_encoder(noisy_trajectory)  # (B, T, dit_dim)
        pos_ids = torch.arange(gt_trajectory.shape[1], device=device).reshape(1, -1).repeat(bsz, 1)
        pos_embed = self.pos_encoding(pos_ids).to(dtype=action_features.dtype)
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
                loss_val = loss.sum() * 0.0
        else:
            loss_val = loss.mean()

        return {"loss": loss_val}

    # ==================== Flow Matching Inference ====================

    def _generate_traj_from_condition_latents(
        self,
        latents_cond: torch.Tensor,
        *,
        predict_step_nums: int,
        guidance_scale: float,
        num_inference_steps: int,
        num_sample_trajs: int,
        generator: torch.Generator | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        device = latents_cond.device
        dtype = latents_cond.dtype
        batch_size = latents_cond.shape[0]

        # Classifier-Free Guidance: [unconditional, conditional]
        hidden_states_null = torch.zeros_like(latents_cond)
        hidden_states_input = torch.cat([hidden_states_null, latents_cond], dim=0)
        hidden_states_input = hidden_states_input.repeat_interleave(num_sample_trajs, dim=0)

        # Initialize random noise
        expected_noise_shape = (
            batch_size * num_sample_trajs,
            predict_step_nums,
            self.config.action_dim,
        )
        if initial_noise is None:
            traj_latents = randn_tensor(
                shape=expected_noise_shape,
                generator=generator,
                device=device,
                dtype=dtype,
            )
        else:
            if tuple(initial_noise.shape) != expected_noise_shape:
                raise ValueError(
                    f"initial_noise must be {expected_noise_shape}, got "
                    f"{tuple(initial_noise.shape)}"
                )
            if not initial_noise.is_floating_point() or not torch.isfinite(
                initial_noise
            ).all():
                raise ValueError("initial_noise must be finite floating point")
            traj_latents = initial_noise.to(device=device, dtype=dtype).clone()

        # Reuse the existing noise scheduler to avoid repeated allocations.
        # Reset timesteps for inference (training config is preserved after this call).
        sigmas = np.linspace(1.0, 1 / num_inference_steps, num_inference_steps)
        self.noise_scheduler.set_timesteps(num_inference_steps, sigmas=sigmas)

        # Iterative denoising
        for t in self.noise_scheduler.timesteps:
            latent_features = self.action_encoder(traj_latents)
            pos_ids = (
                torch.arange(latent_features.shape[1], device=device)
                .reshape(1, -1)
                .repeat(batch_size * num_sample_trajs, 1)
            )
            pos_embed = self.pos_encoding(pos_ids).to(dtype=latent_features.dtype)
            latent_features = latent_features + pos_embed

            # Double for CFG
            latent_model_input = latent_features.repeat(2, 1, 1)
            if hasattr(self.noise_scheduler, "scale_model_input"):
                latent_model_input = self.noise_scheduler.scale_model_input(latent_model_input, t)

            noise_pred = self.traj_dit(
                x=latent_model_input,
                timestep=t.unsqueeze(0).expand(latent_model_input.shape[0]).to(device, torch.long),
                z_latents=hidden_states_input,
            )
            noise_pred = self.action_decoder(noise_pred)

            # CFG interpolation
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)

            traj_latents = self.noise_scheduler.step(noise_pred, t, traj_latents).prev_sample

        return traj_latents

    @torch.no_grad()
    def generate_traj(
        self,
        traj_hidden_states: torch.Tensor,
        traj_images: torch.Tensor | None = None,
        predict_step_nums: int = 32,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 10,
        num_sample_trajs: int = 32,
        generator: torch.Generator | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        Generate trajectory via iterative flow matching denoising with CFG.

        Args:
            traj_hidden_states: (B, n_query, vlm_hidden_dim)
            traj_images: (B, 2, H, W, 3) — [anchor, current]
            predict_step_nums: number of trajectory steps to predict
            guidance_scale: classifier-free guidance scale
            num_inference_steps: number of denoising steps
            num_sample_trajs: number of parallel trajectory samples

        Returns:
            latents: (B * num_sample_trajs, predict_step_nums, action_dim)
        """
        latents_cond = self._fuse_conditions(traj_hidden_states, traj_images)
        return self._generate_traj_from_condition_latents(
            latents_cond,
            predict_step_nums=predict_step_nums,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_sample_trajs=num_sample_trajs,
            generator=generator,
            initial_noise=initial_noise,
        )

    @torch.no_grad()
    def generate_traj_from_projected(
        self,
        traj_cond: torch.Tensor,
        traj_images: torch.Tensor | None = None,
        predict_step_nums: int = 32,
        guidance_scale: float = 1.0,
        num_inference_steps: int = 10,
        num_sample_trajs: int = 32,
        generator: torch.Generator | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Generate trajectories from pre-projected NextDiT condition tokens."""
        latents_cond = self._fuse_projected_conditions(traj_cond, traj_images)
        return self._generate_traj_from_condition_latents(
            latents_cond,
            predict_step_nums=predict_step_nums,
            guidance_scale=guidance_scale,
            num_inference_steps=num_inference_steps,
            num_sample_trajs=num_sample_trajs,
            generator=generator,
            initial_noise=initial_noise,
        )

    @torch.no_grad()
    def get_trajectory(
        self,
        traj_hidden_states: torch.Tensor,
        traj_images: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """
        High-level trajectory inference interface.

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
            generator=generator,
            initial_noise=initial_noise,
        )

    @torch.no_grad()
    def get_trajectory_from_projected(
        self,
        traj_cond: torch.Tensor,
        traj_images: torch.Tensor | None = None,
        *,
        generator: torch.Generator | None = None,
        initial_noise: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """High-level inference interface for direct 768-dim conditions."""
        self.eval()
        return self.generate_traj_from_projected(
            traj_cond=traj_cond,
            traj_images=traj_images,
            predict_step_nums=self.config.predict_steps,
            guidance_scale=self.config.guidance_scale,
            num_inference_steps=self.config.num_inference_steps,
            num_sample_trajs=self.config.num_sample_trajs,
            generator=generator,
            initial_noise=initial_noise,
        )

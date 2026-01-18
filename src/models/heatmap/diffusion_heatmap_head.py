"""
Diffusion Heatmap Head - Main Module
=====================================

This module provides diffusion-based heatmap generation for VLN navigation.
It takes LLM token features and observation images as input, and generates
spatial heatmaps using a denoising diffusion process.

Architecture:
    Input:
        - LLM tokens: (B, seq_len, 2048)
        - Observation: (B, 3, H, W)
    
    Processing:
        1. MultiModalConditionEncoder: Fuse LLM + image -> (B, cond_dim)
        2. ConditionalUnet2D: Predict noise in heatmap
        3. DDPMScheduler: Iterative denoising
    
    Output:
        - Heatmap: (B, Hm, Wm) probability distribution

Usage:
    config = DiffusionHeatmapConfig()
    head = DiffusionHeatmapHead(config).to(device)
    
    # Inference
    heatmap = head(llm_tokens, observation)
    
    # Training
    result = head(llm_tokens, observation, gt_heatmap=gt, return_loss=True)
    loss = result['loss']
"""

import logging
from typing import Optional, Dict, Any, Union, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from .diffusion.config import DiffusionHeatmapConfig
from .diffusion.unet2d import ConditionalUnet2D
from .diffusion.image_encoder import MultiModalConditionEncoder

logger = logging.getLogger(__name__)


class DiffusionHeatmapHead(nn.Module):
    """
    Diffusion-based Heatmap Generation Head.
    
    Generates spatial heatmaps using a denoising diffusion process conditioned
    on LLM tokens and observation images.
    
    Args:
        config: DiffusionHeatmapConfig with all hyperparameters
    
    Attributes:
        condition_encoder: Fuses LLM + image into conditioning vector
        noise_predictor: ConditionalUnet2D for noise prediction
        noise_scheduler: DDPM scheduler for diffusion
    """
    
    def __init__(self, config: DiffusionHeatmapConfig):
        super().__init__()
        
        self.config = config
        self.heatmap_size = config.heatmap_size
        
        # Training optimization: inference monitoring control
        self._training_step_counter = 0
        self._inference_interval = 100  # Generate heatmap every N steps during training
        self._peak_loss_interval = 5    # Compute peak loss every N steps (restored to more frequent)
        
        # ==================== Condition Encoder ====================
        self.condition_encoder = MultiModalConditionEncoder(
            llm_dim=config.llm_dim,
            image_channels=config.image_channels,
            cond_dim=config.cond_dim,
            image_encoder_channels=config.image_encoder_channels,
            llm_hidden_dim=config.llm_hidden_dim,
            pool_method=config.llm_pool_method,
            pool_num_heads=config.llm_pool_num_heads,
            image_size=config.image_size,
            dropout=config.dropout,
            use_image_encoder=config.use_image_encoder,
        )
        
        # ==================== Noise Predictor ====================
        self.noise_predictor = ConditionalUnet2D(
            in_channels=config.in_channels,
            out_channels=config.out_channels,
            cond_dim=config.cond_dim,
            block_out_channels=config.block_out_channels,
            layers_per_block=config.layers_per_block,
            attention_levels=config.attention_levels,
            n_groups=config.norm_num_groups,
            dropout=config.dropout,
            use_circular_padding=config.use_circular_padding,  # 360° 全景图支持
        )
        
        # ==================== Noise Scheduler ====================
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            prediction_type=config.prediction_type,
        )
        
        # Count parameters
        total_params = sum(p.numel() for p in self.parameters())
        logger.info(
            f"DiffusionHeatmapHead initialized: "
            f"heatmap_size={config.heatmap_size}, "
            f"cond_dim={config.cond_dim}, "
            f"use_image_encoder={config.use_image_encoder}, "
            f"pool_method={config.llm_pool_method}, "
            f"params={total_params:,}"
        )
    
    def forward(
        self,
        llm_tokens: torch.Tensor,
        observation: torch.Tensor,
        gt_heatmap: Optional[torch.Tensor] = None,
        return_loss: bool = False,
        skip_inference: bool = False,  # 🆕 训练时跳过推理以提升速度
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for heatmap generation.
        
        Args:
            llm_tokens: (B, ..., llm_dim) LLM features (any shape, will be pooled)
            observation: (B, C, H, W) observation image
            gt_heatmap: Optional (B, Hm, Wm) ground truth heatmap for training
            return_loss: If True and gt_heatmap provided, return loss dict
            skip_inference: If True, skip inference during training for speed
            
        Returns:
            If return_loss and gt_heatmap provided:
                Dict with 'heatmap', 'loss', 'noise_pred', 'noise_target'
            Else:
                (B, Hm, Wm) predicted heatmap
        """
        # 1. Preprocess llm_tokens: flatten intermediate dimensions to (B, seq_len, D)
        # Keep sequence dimension for attention pooling in condition_encoder
        if llm_tokens.dim() > 3:
            # (B, K, seq_len, D) -> (B, K*seq_len, D)
            B = llm_tokens.shape[0]
            D = llm_tokens.shape[-1]
            llm_tokens = llm_tokens.reshape(B, -1, D)
        # Now llm_tokens is (B, seq_len, D) - let condition_encoder handle pooling
        
        # 2. Encode conditions (condition_encoder will pool based on configured pool_method)
        cond = self.condition_encoder(llm_tokens, observation)  # (B, cond_dim)
        
        # 3. Training mode
        if gt_heatmap is not None and return_loss:
            return self._compute_training_loss(cond, gt_heatmap, skip_inference)
        
        # 4. Inference mode
        heatmap = self._diffusion_inference(cond)
        
        if return_loss:
            return {
                'heatmap': heatmap,
                'loss': torch.tensor(0.0, device=heatmap.device),
            }
        
        return heatmap
    
    def _compute_training_loss(
        self,
        cond: torch.Tensor,
        gt_heatmap: torch.Tensor,
        skip_inference: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute diffusion training loss.
        
        Args:
            cond: (B, cond_dim) conditioning vector
            gt_heatmap: (B, Hm, Wm) ground truth heatmap
            skip_inference: If True, skip inference for monitoring (faster training)
            
        Returns:
            Dict with 'loss', 'heatmap', 'noise_pred', 'noise_target'
        """
        device = cond.device
        batch_size = cond.shape[0]
        
        # Prepare GT heatmap: (B, Hm, Wm) -> (B, 1, Hm, Wm)
        if gt_heatmap.dim() == 3:
            gt_heatmap = gt_heatmap.unsqueeze(1)
        
        # Resize GT to target size if needed
        Hm, Wm = self.heatmap_size
        if gt_heatmap.shape[-2:] != (Hm, Wm):
            gt_heatmap = F.interpolate(
                gt_heatmap, size=(Hm, Wm), mode='bilinear', align_corners=False
            )
        
        # Normalize GT heatmap to [-1, 1] for diffusion
        gt_normalized = self._normalize_heatmap(gt_heatmap)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=device, dtype=torch.long
        )
        
        # Sample noise
        noise = torch.randn_like(gt_normalized)
        
        # Add noise to heatmap (forward diffusion)
        noisy_heatmap = self.noise_scheduler.add_noise(gt_normalized, noise, timesteps)
        
        # Predict noise
        noise_pred = self.noise_predictor(
            sample=noisy_heatmap,
            timestep=timesteps,
            global_cond=cond,
        )
        
        # ✅ 标准 DDPM Loss：无加权 MSE
        # 原理：Diffusion 的去噪过程自然会学习数据分布
        #       不需要人为约束（peak_loss/variance_loss 反而有害）
        #       噪声 ε 在空间上均匀分布，加权会破坏这个假设
        diffusion_loss = F.mse_loss(noise_pred, noise)
        
        # 诊断信息：记录噪声预测质量
        noise_std = noise.std().item()
        noise_pred_std = noise_pred.std().item()
        
        # 定期生成预测热力图用于可视化（不参与 loss 计算）
        self._training_step_counter += 1
        pred_heatmap = None
        
        if not skip_inference and (self._training_step_counter % self._inference_interval == 0):
            with torch.no_grad():
                pred_heatmap = self._diffusion_inference(cond)
        
        return {
            'loss': diffusion_loss,
            'diffusion_loss': diffusion_loss,
            'heatmap': pred_heatmap,
            'noise_pred': noise_pred,
            'noise_target': noise,
            'noise_std': noise_std,
            'noise_pred_std': noise_pred_std,
        }
    
    def _diffusion_inference(self, cond: torch.Tensor) -> torch.Tensor:
        """
        Iterative denoising to generate heatmap.
        
        Args:
            cond: (B, cond_dim) conditioning vector
            
        Returns:
            (B, Hm, Wm) predicted heatmap (probability distribution)
        """
        device = cond.device
        batch_size = cond.shape[0]
        Hm, Wm = self.heatmap_size
        
        # Initialize with random noise
        noisy_heatmap = torch.randn(
            (batch_size, 1, Hm, Wm),
            device=device,
            dtype=cond.dtype,
        )
        
        # Set scheduler timesteps
        self.noise_scheduler.set_timesteps(self.config.num_inference_steps)
        
        # Iterative denoising
        for t in self.noise_scheduler.timesteps:
            # Predict noise
            noise_pred = self.noise_predictor(
                sample=noisy_heatmap,
                timestep=t.unsqueeze(-1).repeat(batch_size).to(device),
                global_cond=cond,
            )
            
            # Remove noise (reverse diffusion step)
            noisy_heatmap = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=noisy_heatmap,
            ).prev_sample
        
        # Denormalize and convert to probability
        heatmap = self._denormalize_heatmap(noisy_heatmap)
        
        # Remove channel dimension: (B, 1, Hm, Wm) -> (B, Hm, Wm)
        return heatmap.squeeze(1)
    
    def _normalize_heatmap(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        平方根归一化：缓解稀疏热力图的分布不均匀问题
        
        热力图特点：大部分是 0（背景），少数峰值接近 1
        平方根变换让小值变大，使分布更均匀，同时保持简单
        
        变换: sqrt(x) * 2 - 1
        - x=0 -> sqrt(0)*2-1 = -1
        - x=0.25 -> sqrt(0.25)*2-1 = 0  (中点移动到 0.25)
        - x=1 -> sqrt(1)*2-1 = 1
        
        Args:
            heatmap: (B, 1, H, W) heatmap in [0, 1]
            
        Returns:
            (B, 1, H, W) heatmap in [-1, 1]
        """
        heatmap = heatmap.clamp(0, 1)
        # 平方根变换让小值放大，使分布更均匀
        sqrt_heatmap = torch.sqrt(heatmap)
        # 映射到 [-1, 1]
        return sqrt_heatmap * 2 - 1
    
    def _denormalize_heatmap(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        平方根反归一化：还原热力图
        
        反变换: ((x+1)/2)^2
        
        Args:
            heatmap: (B, 1, H, W) heatmap in [-1, 1]
            
        Returns:
            (B, 1, H, W) heatmap in [0, 1]
        """
        # 先映射到 [0, 1]
        sqrt_recovered = (heatmap + 1) / 2
        sqrt_recovered = sqrt_recovered.clamp(0, 1)
        # 平方还原
        recovered = sqrt_recovered ** 2
        return recovered
    
    def forward_llm_only(
        self,
        llm_tokens: torch.Tensor,
        heatmap_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        """
        Generate heatmap using only LLM tokens (no observation).
        
        Args:
            llm_tokens: (B, seq_len, llm_dim) LLM features
            heatmap_size: Optional output size override
            
        Returns:
            (B, Hm, Wm) predicted heatmap
        """
        cond = self.condition_encoder.forward_llm_only(llm_tokens)
        return self._diffusion_inference(cond)


def create_diffusion_heatmap_head(
    llm_dim: int = 2048,
    cond_dim: int = 512,
    heatmap_size: Tuple[int, int] = (64, 64),
    num_inference_steps: int = 10,
    device: str = "cuda",
    **kwargs,
) -> DiffusionHeatmapHead:
    """
    Factory function to create a DiffusionHeatmapHead.
    
    Args:
        llm_dim: LLM feature dimension
        cond_dim: Conditioning dimension
        heatmap_size: Output heatmap size
        num_inference_steps: Inference diffusion steps
        device: Target device
        **kwargs: Additional config parameters
        
    Returns:
        Initialized DiffusionHeatmapHead
    """
    config = DiffusionHeatmapConfig(
        llm_dim=llm_dim,
        cond_dim=cond_dim,
        heatmap_size=heatmap_size,
        num_inference_steps=num_inference_steps,
        **kwargs,
    )
    
    model = DiffusionHeatmapHead(config)
    model = model.to(device)
    
    return model


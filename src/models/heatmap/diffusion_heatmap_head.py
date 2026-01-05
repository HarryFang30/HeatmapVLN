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
        
        # ==================== Condition Encoder ====================
        self.condition_encoder = MultiModalConditionEncoder(
            llm_dim=config.llm_dim,
            image_channels=config.image_channels,
            cond_dim=config.cond_dim,
            image_encoder_channels=config.image_encoder_channels,
            llm_hidden_dim=config.llm_hidden_dim,
            pool_method=config.llm_pool_method,
            image_size=config.image_size,
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
        # 1. Preprocess llm_tokens to (B, D) by mean pooling
        if llm_tokens.dim() > 2:
            # Pool all intermediate dimensions: (B, K, seq_len, D) -> (B, D)
            # or (B, seq_len, D) -> (B, D)
            B = llm_tokens.shape[0]
            D = llm_tokens.shape[-1]
            llm_tokens = llm_tokens.reshape(B, -1, D).mean(dim=1)  # (B, D)
        
        # 2. Encode conditions
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
        
        # 🔧 [FIX] 加权 MSE Loss：峰值区域权重更高
        # 问题：GT热力图93.5%是黑色，如果用普通MSE，模型学会"输出全黑"就能获得极低Loss
        # 解决：增加峰值区域（热力图高值区域）的权重，强制模型学习空间结构
        
        # 计算权重：峰值区域权重 x10，背景权重 x1
        # gt_heatmap: [B, 1, H, W] in [0, 1]
        weight = 1.0 + 9.0 * gt_heatmap.clamp(0, 1)  # 范围：[1.0, 10.0]
        
        # 加权 MSE
        squared_error = (noise_pred - noise).pow(2)
        weighted_loss = (weight * squared_error).mean()
        
        diffusion_loss = weighted_loss
        
        # 🔍 诊断信息：记录噪声预测质量
        noise_std = noise.std().item()
        noise_pred_std = noise_pred.std().item()
        
        # 🔧 [FIX-2] 峰值保持损失：确保输出热力图有明显峰值，不能全黑
        # 每隔一定步数生成预测热力图用于计算峰值损失
        self._training_step_counter += 1
        pred_heatmap = None
        peak_loss = torch.tensor(0.0, device=device)
        variance_loss = torch.tensor(0.0, device=device)

        # 每3步计算一次峰值保持损失（更频繁以更好地防止坍缩）
        compute_peak_loss = (self._training_step_counter % 3 == 0)

        if compute_peak_loss or not skip_inference:
            with torch.no_grad():
                pred_heatmap = self._diffusion_inference(cond)

            if compute_peak_loss and pred_heatmap is not None:
                # 峰值约束：pred_heatmap.max() 必须 >= 0.3
                # 如果最大值小于0.3，则产生惩罚
                peak_loss = F.relu(0.3 - pred_heatmap.max())

                # 方差约束：输出必须有空间变化（不能全是同一个值）
                # 如果标准差小于0.05，则产生惩罚（提高阈值）
                variance_loss = F.relu(0.05 - pred_heatmap.std())

        # 总损失 = 扩散损失 + 峰值保持损失
        # 增加峰值损失权重以更好地防止过拟合到全黑
        loss = diffusion_loss + 1.0 * (peak_loss + variance_loss)
        
        return {
            'loss': loss,
            'diffusion_loss': diffusion_loss,
            'peak_loss': peak_loss,
            'variance_loss': variance_loss,
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
        对数空间归一化：更好地保留峰值信息
        
        使用对数变换让信号分布更均匀，扩散模型更容易学习。
        不截断小值，保留所有信息。
        
        Args:
            heatmap: (B, 1, H, W) heatmap (非负值)
            
        Returns:
            (B, 1, H, W) heatmap in [-1, 1] (对数空间)
        """
        # 先做 max-to-1 归一化，确保输入范围一致
        B = heatmap.shape[0]
        max_vals = heatmap.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
        max_vals = max_vals.clamp(min=1e-6)  # 仅避免除零
        heatmap_norm = heatmap / max_vals
        
        # 对数变换：使用 log1p 风格的变换更稳定
        # x=0 -> log(1)=0, x=1 -> log(scale+1)
        log_scale = 6.0
        log_heatmap = torch.log(heatmap_norm * log_scale + 1)
        max_log = torch.log(torch.tensor(log_scale + 1, device=heatmap.device, dtype=heatmap.dtype))
        
        # 归一化到 [-1, 1]
        normalized = (log_heatmap / max_log) * 2 - 1
        
        return normalized
    
    def _denormalize_heatmap(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        从对数空间反归一化到热力图
        
        使用 max-to-1 归一化，不截断小值，保持相对比例。
        
        Args:
            heatmap: (B, 1, H, W) heatmap in [-1, 1] (对数空间)
            
        Returns:
            (B, 1, H, W) normalized heatmap (max value = 1)
        """
        log_scale = 6.0
        max_log = torch.log(torch.tensor(log_scale + 1, device=heatmap.device, dtype=heatmap.dtype))
        
        # 从 [-1, 1] 反归一化到 [0, max_log]
        log_heatmap = (heatmap + 1) / 2 * max_log
        
        # 从对数空间恢复：exp(log_heatmap) - 1 然后除以 scale
        recovered = (torch.exp(log_heatmap) - 1) / log_scale
        
        # Max-to-1 归一化（不截断小值，保持相对比例）
        B = heatmap.shape[0]
        max_vals = recovered.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1)
        max_vals = max_vals.clamp(min=1e-6)  # 仅避免除零
        normalized = recovered / max_vals
        
        return normalized
    
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


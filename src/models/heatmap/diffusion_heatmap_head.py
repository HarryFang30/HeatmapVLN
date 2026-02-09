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
        
        # Classifier-Free Guidance (CFG) 参数
        self.cfg_drop_prob = config.cfg_drop_prob
        self.cfg_scale = config.cfg_scale
        
        # Sequence conditioning flag
        self.use_sequence_conditioning = config.use_sequence_conditioning
        
        # Visibility head flag
        self.use_visibility_head = config.use_visibility_head
        self.visibility_threshold = config.visibility_threshold
        self.visibility_loss_weight = config.visibility_loss_weight
        
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
            use_sequence_conditioning=config.use_sequence_conditioning,
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
            # Sequence cross-attention conditioning
            use_sequence_conditioning=config.use_sequence_conditioning,
            seq_cross_attn_heads=config.seq_cross_attn_heads,
            seq_cross_attn_head_dim=config.seq_cross_attn_head_dim,
        )
        
        # ==================== Visibility Head ====================
        # 可见性预测头：判断当前视角是否能看到历史点
        # 当预测为不可见时跳过扩散推理，直接输出全零热力图
        if self.use_visibility_head:
            self.visibility_head = nn.Sequential(
                nn.Linear(config.cond_dim, config.cond_dim // 4),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.cond_dim // 4, 1),
            )
            logger.info("VisibilityHead enabled: cond_dim=%d -> 1 (binary)", config.cond_dim)
        
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
        
        # 2. Encode conditions (dual-path if sequence conditioning enabled)
        if self.use_sequence_conditioning:
            cond, seq_cond = self.condition_encoder.forward_dual(llm_tokens, observation)
        else:
            cond = self.condition_encoder(llm_tokens, observation)  # (B, cond_dim)
            seq_cond = None
        
        # 3. Training mode
        if gt_heatmap is not None and return_loss:
            return self._compute_training_loss(cond, gt_heatmap, skip_inference, seq_cond=seq_cond)
        
        # 4. Inference mode (with visibility gating)
        B = cond.shape[0]
        Hm, Wm = self.heatmap_size
        visibility_score = None
        
        if self.use_visibility_head:
            # 先判可见性，不可见样本直接跳过扩散推理（省时 + 消除假阳性）
            with torch.no_grad():
                vis_logit = self.visibility_head(cond)  # (B, 1)
                visibility_score = torch.sigmoid(vis_logit).squeeze(-1)  # (B,)
                visible_mask = visibility_score >= self.visibility_threshold  # (B,)
            
            if visible_mask.any() and not visible_mask.all():
                # 混合 batch：只对可见样本跑扩散
                heatmap = torch.zeros(B, Hm, Wm, device=cond.device, dtype=cond.dtype)
                visible_cond = cond[visible_mask]
                visible_seq = seq_cond[visible_mask] if seq_cond is not None else None
                heatmap[visible_mask] = self._diffusion_inference(visible_cond, seq_cond=visible_seq)
            elif visible_mask.all():
                # 全部可见：正常跑扩散
                heatmap = self._diffusion_inference(cond, seq_cond=seq_cond)
            else:
                # 全部不可见：直接全零
                heatmap = torch.zeros(B, Hm, Wm, device=cond.device, dtype=cond.dtype)
        else:
            # 无 visibility head：正常跑扩散
            heatmap = self._diffusion_inference(cond, seq_cond=seq_cond)
        
        if return_loss:
            result = {
                'heatmap': heatmap,
                'loss': torch.tensor(0.0, device=heatmap.device),
            }
            if visibility_score is not None:
                result['visibility_score'] = visibility_score
            return result
        
        return heatmap
    
    def _compute_training_loss(
        self,
        cond: torch.Tensor,
        gt_heatmap: torch.Tensor,
        skip_inference: bool = False,
        seq_cond: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute diffusion training loss.
        
        Args:
            cond: (B, cond_dim) conditioning vector
            gt_heatmap: (B, Hm, Wm) ground truth heatmap
            skip_inference: If True, skip inference for monitoring (faster training)
            seq_cond: Optional (B, seq_len, cond_dim) sequence conditioning
            
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
        
        # Classifier-Free Guidance: 训练时随机 drop 条件
        # 让模型学会无条件生成，用于推理时的引导
        if self.training and self.cfg_drop_prob > 0:
            # 创建 drop mask
            drop_mask = torch.rand(batch_size, device=device) < self.cfg_drop_prob
            # 将 drop 的样本条件设为零向量
            cond_for_pred = cond.clone()
            cond_for_pred[drop_mask] = 0.0
            # 序列条件也需要 drop
            seq_cond_for_pred = seq_cond
            if seq_cond is not None:
                seq_cond_for_pred = seq_cond.clone()
                seq_cond_for_pred[drop_mask] = 0.0
        else:
            cond_for_pred = cond
            seq_cond_for_pred = seq_cond
        
        # Predict noise
        noise_pred = self.noise_predictor(
            sample=noisy_heatmap,
            timestep=timesteps,
            global_cond=cond_for_pred,
            seq_cond=seq_cond_for_pred,
        )
        
        # ==================== Focal-style 混合损失 + 负样本降权 ====================
        with torch.no_grad():
            # 检测每个样本是否为负样本（GT 全零，即不可见目标）
            # gt_heatmap: (B, 1, H, W), 按样本维度 flatten 后取 max
            sample_max = gt_heatmap.flatten(1).max(dim=1).values  # (B,)
            is_negative = (sample_max < 0.01).float()  # (B,)
            
            # 负样本扩散 loss 权重降低到 0.1（避免扩散头学习"恢复全零"）
            # 正样本权重 1.0，负样本权重 0.1
            sample_weight = 1.0 - 0.9 * is_negative  # (B,) = 1.0 or 0.1
            # 重塑为 (B, 1, 1, 1) 以广播到空间维度
            sample_weight = sample_weight.view(-1, 1, 1, 1)
        
        # 逐样本 MSE（不做 reduction）
        per_pixel_mse = (noise_pred - noise) ** 2  # (B, 1, H, W)
        
        # 主损失：标准 MSE（无空间加权，保持 DDPM 理论正确性）+ 样本级负样本降权
        base_loss = (sample_weight * per_pixel_mse).mean()
        
        # Focal 损失：让模型更关注峰值区域的噪声预测
        with torch.no_grad():
            gt_weight = gt_heatmap.clamp(0, 1)
            focal_alpha = 1.0
            weight_map = 1.0 + focal_alpha * gt_weight
            weight_map = weight_map / weight_map.mean()
        
        focal_loss = (sample_weight * weight_map * per_pixel_mse).mean()
        
        # 混合损失
        focal_weight = 0.3
        diffusion_loss = (1 - focal_weight) * base_loss + focal_weight * focal_loss
        
        total_loss = diffusion_loss
        
        # ==================== Visibility Loss ====================
        # 可见性预测：判断当前样本的 GT 热力图是否有峰值
        visibility_loss_val = 0.0
        if self.use_visibility_head:
            # GT 标签：热力图最大值 > 0.01 为"可见"
            with torch.no_grad():
                gt_has_peak = (gt_heatmap.flatten(1).max(dim=1).values > 0.01).float()  # (B,)
            
            vis_logit = self.visibility_head(cond).squeeze(-1)  # (B,)
            visibility_loss = F.binary_cross_entropy_with_logits(vis_logit, gt_has_peak)
            visibility_loss_val = visibility_loss.item()
            
            total_loss = total_loss + self.visibility_loss_weight * visibility_loss
        
        # 诊断信息：记录噪声预测质量
        noise_std = noise.std().item()
        noise_pred_std = noise_pred.std().item()
        
        # 定期生成预测热力图用于可视化（不参与 loss 计算）
        self._training_step_counter += 1
        pred_heatmap = None
        
        if not skip_inference and (self._training_step_counter % self._inference_interval == 0):
            with torch.no_grad():
                pred_heatmap = self._diffusion_inference(cond, seq_cond=seq_cond)
        
        return {
            'loss': total_loss,
            'diffusion_loss': diffusion_loss,
            'base_loss': base_loss.item(),          # 标准 MSE
            'focal_loss': focal_loss.item(),        # 峰值加权 MSE
            'visibility_loss': visibility_loss_val, # 可见性 BCE loss
            'heatmap': pred_heatmap,
            'noise_pred': noise_pred,
            'noise_target': noise,
            'noise_std': noise_std,
            'noise_pred_std': noise_pred_std,
        }
    
    def _diffusion_inference(
        self,
        cond: torch.Tensor,
        use_cfg: bool = True,
        seq_cond: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Iterative denoising to generate heatmap with Classifier-Free Guidance.
        
        Args:
            cond: (B, cond_dim) conditioning vector
            use_cfg: 是否使用 CFG 引导（推理时建议开启）
            seq_cond: Optional (B, seq_len, cond_dim) sequence conditioning
            
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
        
        # 准备无条件向量（零向量）
        uncond = torch.zeros_like(cond)
        uncond_seq = torch.zeros_like(seq_cond) if seq_cond is not None else None
        
        # 是否使用 CFG
        do_cfg = use_cfg and self.cfg_scale > 1.0
        
        # Set scheduler timesteps
        self.noise_scheduler.set_timesteps(self.config.num_inference_steps)
        
        # Iterative denoising
        for t in self.noise_scheduler.timesteps:
            timestep_batch = t.unsqueeze(-1).repeat(batch_size).to(device)
            
            if do_cfg:
                # CFG: 同时预测有条件和无条件的噪声
                # 有条件预测
                noise_pred_cond = self.noise_predictor(
                    sample=noisy_heatmap,
                    timestep=timestep_batch,
                    global_cond=cond,
                    seq_cond=seq_cond,
                )
                # 无条件预测
                noise_pred_uncond = self.noise_predictor(
                    sample=noisy_heatmap,
                    timestep=timestep_batch,
                    global_cond=uncond,
                    seq_cond=uncond_seq,
                )
                # CFG 公式：noise = uncond + scale * (cond - uncond)
                noise_pred = noise_pred_uncond + self.cfg_scale * (noise_pred_cond - noise_pred_uncond)
            else:
                # 不使用 CFG
                noise_pred = self.noise_predictor(
                    sample=noisy_heatmap,
                    timestep=timestep_batch,
                    global_cond=cond,
                    seq_cond=seq_cond,
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
        线性归一化：将 [0, 1] 热力图映射到 [-1, 1]
        
        使用线性变换保持误差在所有值域上均匀分布，避免 sqrt 变换导致的：
        1. 峰值压缩（反变换时小误差被放大）
        2. 背景弥散（sqrt 让接近 0 的值被显著放大，模型难以区分纯背景和微弱信号）
        
        变换: x * 2 - 1
        - x=0   -> -1
        - x=0.5 -> 0
        - x=1   -> 1
        
        Args:
            heatmap: (B, 1, H, W) heatmap in [0, 1]
            
        Returns:
            (B, 1, H, W) heatmap in [-1, 1]
        """
        heatmap = heatmap.clamp(0, 1)
        return heatmap * 2 - 1
    
    def _denormalize_heatmap(self, heatmap: torch.Tensor) -> torch.Tensor:
        """
        线性反归一化：将 [-1, 1] 还原到 [0, 1]
        
        反变换: (x + 1) / 2
        
        Args:
            heatmap: (B, 1, H, W) heatmap in [-1, 1]
            
        Returns:
            (B, 1, H, W) heatmap in [0, 1]
        """
        recovered = (heatmap + 1) / 2
        return recovered.clamp(0, 1)
    
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


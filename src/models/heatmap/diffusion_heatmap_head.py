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
from diffusers.schedulers.scheduling_ddim import DDIMScheduler

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
            image_encoder_use_pretrained=config.image_encoder_use_pretrained,
        )
        
        # Spatial injection flag
        self.use_spatial_injection = config.use_spatial_injection
        
        # ==================== Noise Predictor ====================
        # If spatial injection enabled, pass CNN encoder channel dims to UNet
        # When using pretrained ResNet-18, channels are fixed to [64, 128, 256, 512]
        spatial_injection_channels = None
        if config.use_spatial_injection and config.use_image_encoder:
            if config.image_encoder_use_pretrained:
                from .diffusion.image_encoder import ResNetImageConditionEncoder
                spatial_injection_channels = list(ResNetImageConditionEncoder.CHANNELS)
            else:
                spatial_injection_channels = config.image_encoder_channels
            logger.info(f"SpatialInjection enabled: CNN channels {spatial_injection_channels} -> UNet skips")
        
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
            # Spatial feature injection from CNN encoder
            spatial_injection_channels=spatial_injection_channels,
        )
        
        # ==================== Visibility Head ====================
        # 可见性预测头：判断当前视角是否能看到历史点
        # 当预测为不可见时跳过扩散推理，直接输出全零热力图
        # 增强版：3 层 MLP，增加容量以更好区分正/负样本
        if self.use_visibility_head:
            self.visibility_head = nn.Sequential(
                nn.Linear(config.cond_dim, config.cond_dim // 2),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.cond_dim // 2, config.cond_dim // 4),
                nn.GELU(),
                nn.Dropout(config.dropout),
                nn.Linear(config.cond_dim // 4, 1),
            )
            logger.info("VisibilityHead (enhanced 3-layer) enabled: cond_dim=%d -> 1 (binary)", config.cond_dim)
        
        # ==================== Direction Embedding ====================
        # 多视角模式下为 front/right/back/left 注入可学习方向嵌入
        # 零初始化：起步等价于无嵌入，作为残差逐步学习方向特征
        self.direction_embedding = nn.Embedding(config.num_directions, config.cond_dim)
        nn.init.zeros_(self.direction_embedding.weight)
        logger.info("DirectionEmbedding: %d directions x %d dim (zero-init)", config.num_directions, config.cond_dim)
        
        # ==================== Noise Scheduler ====================
        # Training: DDPM (stochastic, standard for training)
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            prediction_type=config.prediction_type,
        )
        # Inference: DDIM (deterministic, sharper output than stochastic DDPM)
        # Uses same trained model, no retraining needed
        self.inference_scheduler = DDIMScheduler(
            num_train_timesteps=config.num_train_timesteps,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            prediction_type=config.prediction_type,
        )
        
        # Sharpening config
        self.sharpen_temperature = config.sharpen_temperature
        
        # x0 loss config
        self.x0_loss_weight = config.x0_loss_weight
        self.sparsity_loss_weight = config.sparsity_loss_weight
        
        # Negative sample config
        self.negative_sample_weight = config.negative_sample_weight
        
        # Peak distance loss config
        self.peak_distance_loss_weight = config.peak_distance_loss_weight
        self.peak_loss_temperature = config.peak_loss_temperature
        
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
        skip_inference: bool = False,
        direction_indices: Optional[torch.Tensor] = None,
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
        
        # 2. Encode conditions (with spatial features if enabled)
        spatial_features = None
        if self.use_spatial_injection:
            # Get global + sequence + spatial features in one pass
            cond, seq_cond, spatial_features = self.condition_encoder.forward_with_spatial(
                llm_tokens, observation
            )
        elif self.use_sequence_conditioning:
            cond, seq_cond = self.condition_encoder.forward_dual(llm_tokens, observation)
        else:
            cond = self.condition_encoder(llm_tokens, observation)  # (B, cond_dim)
            seq_cond = None
        
        # 2.5 Inject direction embedding for multi-view differentiation
        if direction_indices is not None:
            dir_emb = self.direction_embedding(direction_indices)  # (B, cond_dim)
            cond = cond + dir_emb
            if seq_cond is not None:
                seq_cond = seq_cond + dir_emb.unsqueeze(1)
        
        # 3. Training mode
        if gt_heatmap is not None and return_loss:
            return self._compute_training_loss(
                cond, gt_heatmap, skip_inference, 
                seq_cond=seq_cond, spatial_features=spatial_features,
            )
        
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
                visible_spatial = None
                if spatial_features is not None:
                    visible_spatial = [f[visible_mask] for f in spatial_features]
                heatmap[visible_mask] = self._diffusion_inference(
                    visible_cond, seq_cond=visible_seq, spatial_features=visible_spatial,
                )
            elif visible_mask.all():
                # 全部可见：正常跑扩散
                heatmap = self._diffusion_inference(
                    cond, seq_cond=seq_cond, spatial_features=spatial_features,
                )
            else:
                # 全部不可见：直接全零
                heatmap = torch.zeros(B, Hm, Wm, device=cond.device, dtype=cond.dtype)
        else:
            # 无 visibility head：正常跑扩散
            heatmap = self._diffusion_inference(
                cond, seq_cond=seq_cond, spatial_features=spatial_features,
            )
        
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
        spatial_features: Optional[list] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute diffusion training loss.
        
        Args:
            cond: (B, cond_dim) conditioning vector
            gt_heatmap: (B, Hm, Wm) ground truth heatmap
            skip_inference: If True, skip inference for monitoring (faster training)
            seq_cond: Optional (B, seq_len, cond_dim) sequence conditioning
            spatial_features: Optional CNN multi-scale feature maps for spatial injection
            
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
        
        # Predict noise (with spatial features if available)
        # Note: spatial_features use original cond (not CFG-dropped), since spatial info
        # is from the observation image and should always be available
        noise_pred = self.noise_predictor(
            sample=noisy_heatmap,
            timestep=timesteps,
            global_cond=cond_for_pred,
            seq_cond=seq_cond_for_pred,
            spatial_features=spatial_features,
        )
        
        # ==================== Focal-style 混合损失 + 负样本降权 ====================
        with torch.no_grad():
            # 检测每个样本是否为负样本（GT 全零，即不可见目标）
            # gt_heatmap: (B, 1, H, W), 按样本维度 flatten 后取 max
            sample_max = gt_heatmap.flatten(1).max(dim=1).values  # (B,)
            is_negative = (sample_max < 0.01).float()  # (B,)
            
            # 负样本扩散 loss 权重（可配置，默认 0.3，之前是 0.1）
            sample_weight = 1.0 - (1.0 - self.negative_sample_weight) * is_negative  # (B,)
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
        
        # ==================== B1: x0 重构损失 ====================
        # 从噪声预测反推 x0 估计，直接监督输出质量
        # epsilon loss 只关心噪声预测是否准确，x0 loss 直接关心重构热力图是否像 GT
        x0_loss_val = 0.0
        sparsity_loss_val = 0.0
        neg_zero_loss_val = 0.0
        x0_hat = None
        
        # 需要 x0 估计的条件：x0 loss、sparsity loss、或 peak distance loss
        need_x0 = (self.x0_loss_weight > 0 or self.sparsity_loss_weight > 0 
                    or self.peak_distance_loss_weight > 0)
        
        if need_x0:
            with torch.no_grad():
                alpha_bar = self.noise_scheduler.alphas_cumprod.to(device)[timesteps]
                alpha_bar = alpha_bar.view(-1, 1, 1, 1)
            
            # x0 估计: x0 = (noisy - sqrt(1-alpha_bar) * noise_pred) / sqrt(alpha_bar)
            x0_hat = (noisy_heatmap - torch.sqrt(1 - alpha_bar) * noise_pred) / torch.sqrt(alpha_bar)
            x0_hat = x0_hat.clamp(-1, 1)
            
            if self.x0_loss_weight > 0:
                # x0 重构损失：直接监督重构质量（带样本权重）
                # 注意：用 w × MSE 而非 MSE(w*x, w*y)，后者会导致权重变成 w²
                per_pixel_x0_mse = (x0_hat - gt_normalized) ** 2  # (B, 1, H, W)
                x0_loss = (sample_weight * per_pixel_x0_mse).mean()
                x0_loss_val = x0_loss.item()
                total_loss = total_loss + self.x0_loss_weight * x0_loss
            
            # ==================== B2: L1 稀疏性正则化 ====================
            # 鼓励 x0 估计中大部分像素为 0（在 [0,1] 空间中）
            if self.sparsity_loss_weight > 0:
                x0_denorm = (x0_hat + 1) / 2  # 还原到 [0, 1]
                sparsity_loss = x0_denorm.mean()  # 鼓励整体值接近 0
                sparsity_loss_val = sparsity_loss.item()
                total_loss = total_loss + self.sparsity_loss_weight * sparsity_loss
            
            # ==================== B4: 负样本显式零目标损失（SNR 门控） ====================
            # 对负样本（GT=全零），额外约束 x0 应为全 -1（即 [0,1] 空间中的全 0）
            # 同样使用 SNR 门控：高时间步的 x0_hat 接近随机噪声，强加零约束会产生噪声梯度
            if is_negative.any():
                neg_mask = is_negative.bool()
                
                # SNR 门控：只对 x0 估计质量好的样本施加零约束
                with torch.no_grad():
                    alpha_flat_neg = alpha_bar.view(batch_size)
                    snr_neg = alpha_flat_neg / (1 - alpha_flat_neg + 1e-8)
                    high_snr_neg = neg_mask & (snr_neg > 1.0)
                
                if high_snr_neg.any():
                    neg_x0 = x0_hat[high_snr_neg]
                    neg_target = torch.full_like(neg_x0, -1.0)  # [-1,1] 空间中，全零 = -1
                    neg_zero_loss = F.mse_loss(neg_x0, neg_target)
                    neg_zero_loss_val = neg_zero_loss.item()
                    total_loss = total_loss + 0.5 * neg_zero_loss
        
        # ==================== Peak Distance Loss (Multi-Peak Aware, SNR-gated) ====================
        # 多峰感知的可微分峰值距离损失：
        # 1. NMS 检测 GT 中的所有峰值（非微分）
        # 2. 对每个 GT 峰值创建高斯注意力窗口
        # 3. 在窗口内对预测 x0 做 soft-argmax → 提取对应预测位置
        # 4. 计算每峰的 L2 距离并平均
        #
        # SNR 门控：只在 x0 估计质量足够好时（低时间步，高 SNR）计算
        # 高时间步的 x0_hat 接近随机噪声，peak 位置无意义，梯度有害
        peak_dist_loss_val = 0.0
        
        if self.peak_distance_loss_weight > 0 and x0_hat is not None:
            with torch.no_grad():
                # SNR = alpha_bar / (1 - alpha_bar)
                # 只对 SNR > 1 的样本（即 alpha_bar > 0.5，约 t < T/2）计算
                # 这些样本的 x0 估计信噪比足够高，peak 位置有意义
                alpha_flat = alpha_bar.view(batch_size)  # (B,) 安全的 reshape
                snr = alpha_flat / (1 - alpha_flat + 1e-8)  # (B,)
                snr_mask = (snr > 1.0)  # alpha_bar > 0.5
            
            if snr_mask.any():
                # 只对高 SNR 样本计算 peak loss
                snr_x0 = x0_hat[snr_mask]
                snr_gt = gt_heatmap[snr_mask]
                snr_is_neg = is_negative[snr_mask]
                
                peak_dist_loss, peak_dist_loss_val = self._multi_peak_distance_loss(
                    snr_x0, snr_gt, snr_is_neg
                )
                if peak_dist_loss.requires_grad:
                    total_loss = total_loss + self.peak_distance_loss_weight * peak_dist_loss
        
        # ==================== Visibility Loss (B3: 增强权重) ====================
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
                pred_heatmap = self._diffusion_inference(
                    cond, seq_cond=seq_cond, spatial_features=spatial_features,
                )
        
        return {
            'loss': total_loss,
            'diffusion_loss': diffusion_loss,
            'base_loss': base_loss.item(),          # 标准 MSE
            'focal_loss': focal_loss.item(),        # 峰值加权 MSE
            'visibility_loss': visibility_loss_val, # 可见性 BCE loss
            'x0_loss': x0_loss_val,                 # x0 重构损失
            'sparsity_loss': sparsity_loss_val,     # L1 稀疏性损失
            'neg_zero_loss': neg_zero_loss_val,     # 负样本零目标损失
            'peak_dist_loss': peak_dist_loss_val,   # 可微分峰值距离损失
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
        spatial_features: Optional[list] = None,
    ) -> torch.Tensor:
        """
        Iterative denoising to generate heatmap with Classifier-Free Guidance.
        
        Args:
            cond: (B, cond_dim) conditioning vector
            use_cfg: 是否使用 CFG 引导（推理时建议开启）
            seq_cond: Optional (B, seq_len, cond_dim) sequence conditioning
            spatial_features: Optional CNN multi-scale feature maps for spatial injection
            
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
        
        # Set scheduler timesteps (use DDIM for sharper, deterministic inference)
        self.inference_scheduler.set_timesteps(self.config.num_inference_steps)
        
        # Iterative denoising
        for t in self.inference_scheduler.timesteps:
            timestep_batch = t.unsqueeze(-1).repeat(batch_size).to(device)
            
            if do_cfg:
                # CFG: 同时预测有条件和无条件的噪声
                # 有条件预测（spatial_features 始终传递，因为它来自观察图像）
                noise_pred_cond = self.noise_predictor(
                    sample=noisy_heatmap,
                    timestep=timestep_batch,
                    global_cond=cond,
                    seq_cond=seq_cond,
                    spatial_features=spatial_features,
                )
                # 无条件预测（不传 spatial_features，保持 CFG 的"无条件"语义）
                noise_pred_uncond = self.noise_predictor(
                    sample=noisy_heatmap,
                    timestep=timestep_batch,
                    global_cond=uncond,
                    seq_cond=uncond_seq,
                    spatial_features=None,
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
                    spatial_features=spatial_features,
                )
            
            # Remove noise (reverse diffusion step) - DDIM is deterministic
            noisy_heatmap = self.inference_scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=noisy_heatmap,
            ).prev_sample
        
        # Denormalize and convert to probability
        heatmap = self._denormalize_heatmap(noisy_heatmap)
        
        # Remove channel dimension: (B, 1, Hm, Wm) -> (B, Hm, Wm)
        heatmap = heatmap.squeeze(1)
        
        # Post-denoising sharpening: spatial softmax with temperature
        if self.sharpen_temperature > 0:
            heatmap = self._sharpen_heatmap(heatmap, self.sharpen_temperature)
        
        return heatmap
    
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
    
    def _sharpen_heatmap(self, heatmap: torch.Tensor, temperature: float = 0.1) -> torch.Tensor:
        """
        多峰安全的锐化后处理：抑制背景噪声，保留所有峰值。
        
        旧方法（全局 softmax）会指数级放大峰间差异，把弱峰压成 0：
          exp(0.8/0.1) / exp(0.5/0.1) = 2981/148 = 20:1（原始比仅 1.6:1）
        
        新方法：自适应阈值 + ReLU 背景抑制
        1. 对每个样本计算自适应阈值 = max * temperature
        2. ReLU(x - threshold) 抑制背景，保留所有超过阈值的峰
        3. 归一化到 [0, 1]，保持峰间高度比不变
        
        Args:
            heatmap: (B, H, W) heatmap in [0, 1]
            temperature: 阈值比例（0.1 = 低于 max 的 10% 被清零）
            
        Returns:
            (B, H, W) sharpened heatmap in [0, 1]，多峰结构完整保留
        """
        B, H, W = heatmap.shape
        
        # 每样本自适应阈值 = max * temperature
        max_vals = heatmap.flatten(1).max(dim=1).values  # (B,)
        thresholds = (max_vals * temperature).view(B, 1, 1)  # (B, 1, 1)
        
        # ReLU 背景抑制：低于阈值的像素清零，保留所有峰
        sharpened = F.relu(heatmap - thresholds)
        
        # 归一化到 [0, 1]：保持峰间高度比不变
        new_max = sharpened.flatten(1).max(dim=1).values.view(B, 1, 1)  # (B, 1, 1)
        sharpened = sharpened / (new_max + 1e-8)
        
        return sharpened
    
    def _find_gt_peaks(self, gt_heatmap: torch.Tensor, min_value: float = 0.1, nms_kernel: int = 5) -> list:
        """
        非极大值抑制（NMS）检测 GT 热力图中的所有峰值位置。
        
        使用 max_pool2d 进行局部极大值检测：
        一个像素如果等于其 nms_kernel × nms_kernel 邻域内的最大值，
        且值大于 min_value，则被视为峰值。
        
        Args:
            gt_heatmap: (B, 1, H, W) GT heatmap in [0, 1]
            min_value: 最小峰值阈值
            nms_kernel: NMS 窗口大小
            
        Returns:
            List[Tensor]: 每个样本的峰值坐标列表，每个元素 shape (num_peaks, 2) = [(y, x), ...]
                          如果无峰值则为空 tensor
        """
        B, _, H, W = gt_heatmap.shape
        pad = nms_kernel // 2
        
        # Max pooling NMS
        gt_padded = F.pad(gt_heatmap, [pad] * 4, mode='replicate')
        local_max = F.max_pool2d(gt_padded, kernel_size=nms_kernel, stride=1, padding=0)
        
        # 峰值 = 局部最大值 且 大于阈值
        is_peak = (gt_heatmap == local_max) & (gt_heatmap > min_value)  # (B, 1, H, W)
        is_peak = is_peak.squeeze(1)  # (B, H, W)
        
        peaks_per_sample = []
        for i in range(B):
            peak_coords = is_peak[i].nonzero(as_tuple=False)  # (num_peaks, 2)
            
            # 限制每个样本最多 8 个峰值，按 GT 值排序保留最强的
            if len(peak_coords) > 8:
                gt_vals = gt_heatmap[i, 0, peak_coords[:, 0], peak_coords[:, 1]]
                top_k_idx = gt_vals.topk(8).indices
                peak_coords = peak_coords[top_k_idx]
            
            peaks_per_sample.append(peak_coords)
        
        return peaks_per_sample
    
    def _multi_peak_distance_loss(
        self,
        x0_hat: torch.Tensor,
        gt_heatmap: torch.Tensor,
        is_negative: torch.Tensor,
    ) -> Tuple[torch.Tensor, float]:
        """
        多峰感知的可微分峰值距离损失。
        
        处理流程：
        1. 用 NMS 在 GT 上检测所有峰值位置（非微分，仅获取坐标）
        2. 对每个 GT 峰值，创建高斯注意力窗口（sigma=5, 覆盖 ~15px）
        3. 用窗口加权预测 x0，在窗口区域内做 soft-argmax → 预测位置
        4. 计算每个峰值的 L2 距离并平均
        5. 归一化到 [0, 1]
        
        向量化实现：所有峰值打包为一个 batch 一次性计算 soft-argmax。
        
        Args:
            x0_hat: (B, 1, H, W) 预测的 x0 估计，[-1, 1] 空间
            gt_heatmap: (B, 1, H, W) GT 热力图，[0, 1] 空间
            is_negative: (B,) float tensor，1.0 = 负样本
            
        Returns:
            (loss_tensor, loss_value): 可微分的 loss tensor 和 float 值
        """
        device = x0_hat.device
        dtype = x0_hat.dtype
        B, _, H, W = x0_hat.shape
        diag = (H ** 2 + W ** 2) ** 0.5
        
        # Step 1: 只处理正样本
        pos_mask = ~is_negative.bool()
        if not pos_mask.any():
            return torch.tensor(0.0, device=device, requires_grad=False), 0.0
        
        pos_indices = pos_mask.nonzero(as_tuple=False).squeeze(1)  # (N_pos,)
        pos_gt = gt_heatmap[pos_indices]    # (N_pos, 1, H, W)
        pos_x0 = x0_hat[pos_indices]        # (N_pos, 1, H, W)
        
        # Step 2: 检测所有正样本中的 GT 峰值
        with torch.no_grad():
            peaks_per_sample = self._find_gt_peaks(pos_gt)
        
        # 收集所有峰值 → 向量化处理
        # 记录: (样本在 pos 中的 index, peak_y, peak_x)
        sample_ids = []
        peak_ys = []
        peak_xs = []
        
        for i, peaks in enumerate(peaks_per_sample):
            if len(peaks) == 0:
                continue
            for j in range(len(peaks)):
                sample_ids.append(i)
                peak_ys.append(peaks[j, 0].float())
                peak_xs.append(peaks[j, 1].float())
        
        if len(sample_ids) == 0:
            return torch.tensor(0.0, device=device, requires_grad=False), 0.0
        
        K = len(sample_ids)  # 总峰值数
        sample_ids_t = torch.tensor(sample_ids, device=device, dtype=torch.long)
        peak_ys_t = torch.stack(peak_ys)  # (K,)
        peak_xs_t = torch.stack(peak_xs)  # (K,)
        
        # Step 3: 创建 K 个高斯注意力窗口（向量化）
        # window[k, y, x] = exp(-((y - peak_y_k)^2 + (x - peak_x_k)^2) / (2 * sigma^2))
        window_sigma = 5.0  # 覆盖 GT blob (sigma~3) 及其周围区域
        
        y_grid = torch.arange(H, device=device, dtype=dtype).view(1, H, 1)  # (1, H, 1)
        x_grid = torch.arange(W, device=device, dtype=dtype).view(1, 1, W)  # (1, 1, W)
        
        # (K, 1, 1) broadcast with (1, H, 1) and (1, 1, W) → (K, H, W)
        dy = y_grid - peak_ys_t.view(K, 1, 1)  # (K, H, W)
        dx = x_grid - peak_xs_t.view(K, 1, 1)  # (K, H, W)
        windows = torch.exp(-(dy ** 2 + dx ** 2) / (2 * window_sigma ** 2))  # (K, H, W)
        
        # Step 4: 提取每个峰值对应样本的预测 x0
        pred_for_peaks = pos_x0[sample_ids_t, 0]  # (K, H, W)
        
        # 反归一化到 [0, 1] 让 softmax 更自然（-1 背景 → 0，+1 峰值 → 1）
        pred_denorm = (pred_for_peaks + 1) / 2  # (K, H, W)
        
        # Step 5: 窗口加权预测 + soft-argmax（向量化）
        windowed_pred = pred_denorm * windows  # (K, H, W)
        
        flat = windowed_pred.view(K, -1)  # (K, H*W)
        weights = F.softmax(flat / self.peak_loss_temperature, dim=-1).view(K, H, W)
        
        # 坐标期望值
        pred_peak_y = (weights * y_grid.expand(K, H, W)).sum(dim=[1, 2])  # (K,)
        pred_peak_x = (weights * x_grid.expand(K, H, W)).sum(dim=[1, 2])  # (K,)
        
        # Step 6: L2 距离，归一化到 [0, 1]
        dists = ((pred_peak_y - peak_ys_t) ** 2 + (pred_peak_x - peak_xs_t) ** 2).sqrt()
        peak_dist_loss = (dists / diag).mean()
        
        return peak_dist_loss, peak_dist_loss.item()
    
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


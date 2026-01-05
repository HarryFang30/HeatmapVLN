"""
Diffusion Action Head for VLN Navigation

This module provides the main action generation head using diffusion policy.
It takes LLM output features as conditioning and generates 2D navigation actions.

Architecture:
    LLM Features (B, seq_len, 2048) 
        -> Condition Projection (B, 768)
        -> ConditionalUnet1D (noise prediction)
        -> DDPM Denoising Loop
        -> Actions (B, pred_horizon, action_dim)
"""

import logging
from typing import Optional, Dict, Any, Union
import torch
import torch.nn as nn

from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

from .diffusion import ConditionalUnet1D
from .action_config import DiffusionActionConfig
from .utils import ActionStats, unnormalize_actions, get_action_from_diffusion_output

logger = logging.getLogger(__name__)


class ConditionProjector(nn.Module):
    """
    Projects LLM features to conditioning dimension for diffusion.

    Handles variable-length sequences by pooling or selecting.

    Args:
        input_dim: Input dimension from LLM
        output_dim: Output conditioning dimension
        pool_method: How to aggregate sequence ('mean', 'first', 'last')
    """

    def __init__(
        self,
        input_dim: int = 2048,
        output_dim: int = 768,
        pool_method: str = 'mean',
        dropout: float = 0.2,  # 增加默认dropout
    ):
        super().__init__()

        self.pool_method = pool_method
        self.projector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),  # 增加第二层dropout
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (B, seq_len, input_dim) or (B, input_dim)
            
        Returns:
            (B, output_dim) conditioning vector
        """
        # Handle 2D input (already pooled)
        if x.dim() == 2:
            return self.projector(x)
        
        # Pool sequence dimension
        if self.pool_method == 'mean':
            x = x.mean(dim=1)
        elif self.pool_method == 'first':
            x = x[:, 0]
        elif self.pool_method == 'last':
            x = x[:, -1]
        else:
            raise ValueError(f"Unknown pool method: {self.pool_method}")
        
        return self.projector(x)


class DiffusionActionHead(nn.Module):
    """
    Diffusion Policy Action Head for VLN Navigation.
    
    Takes LLM output features as conditioning and generates navigation actions
    using a diffusion-based generative model.
    
    Architecture:
    1. Condition Projector: LLM features -> conditioning embedding
    2. Noise Predictor: ConditionalUnet1D predicts noise at each timestep
    3. DDPM Scheduler: Iterative denoising to generate clean actions
    
    Args:
        config: DiffusionActionConfig with all hyperparameters
        
    Usage:
        config = DiffusionActionConfig()
        action_head = DiffusionActionHead(config).to(device)
        
        # During inference
        llm_features = model.get_llm_features(...)  # (B, seq_len, 2048)
        actions = action_head(llm_features)  # (B, pred_horizon, action_dim)
        
        # During training
        actions, loss = action_head(llm_features, gt_actions=gt_actions)
    """
    
    def __init__(self, config: DiffusionActionConfig):
        super().__init__()
        
        self.config = config
        self.action_dim = config.action_dim
        self.pred_horizon = config.pred_horizon
        
        # Training optimization: inference frequency control
        self._training_step_counter = 0
        self._inference_interval = config.inference_interval
        
        # Action statistics for normalization
        self.action_stats = ActionStats(
            min=config.action_stats_min,
            max=config.action_stats_max
        )
        
        # Condition projector: LLM features -> conditioning
        self.condition_projector = ConditionProjector(
            input_dim=config.cond_dim,
            output_dim=config.encoding_size,
            pool_method='mean'
        )
        
        # Noise prediction network
        self.noise_pred_net = ConditionalUnet1D(
            input_dim=config.action_dim,
            global_cond_dim=config.encoding_size,
            diffusion_step_embed_dim=config.diffusion_step_embed_dim,
            down_dims=config.down_dims,
            kernel_size=config.kernel_size,
            n_groups=config.n_groups,
            cond_predict_scale=config.cond_predict_scale,
        )
        
        # DDPM noise scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=config.num_diffusion_iters,
            beta_schedule=config.beta_schedule,
            clip_sample=config.clip_sample,
            prediction_type=config.prediction_type,
        )
        
        logger.info(
            f"DiffusionActionHead initialized: action_dim={config.action_dim}, "
            f"pred_horizon={config.pred_horizon}, num_diffusion_iters={config.num_diffusion_iters}"
        )
    
    def forward(
        self,
        llm_features: torch.Tensor,
        gt_actions: Optional[torch.Tensor] = None,
        action_valid: Optional[torch.Tensor] = None,  # 🆕 添加 mask 参数
        return_loss: bool = False,
        cumulative: bool = True,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for action generation.
        
        Args:
            llm_features: (B, seq_len, cond_dim) or (B, cond_dim) LLM output features
            gt_actions: Optional (B, pred_horizon, action_dim) ground truth actions for training
            action_valid: Optional (B,) mask indicating which samples have valid actions
            return_loss: If True and gt_actions provided, return loss dict
            cumulative: If True, return cumulative positions; else return deltas
            
        Returns:
            If return_loss and gt_actions provided:
                Dict with 'actions', 'loss', 'normalized_actions'
            Else:
                (B, pred_horizon, action_dim) predicted actions
        """
        batch_size = llm_features.shape[0]
        device = llm_features.device
        
        # 1. Project LLM features to conditioning
        global_cond = self.condition_projector(llm_features)  # (B, encoding_size)
        
        # 2. Training mode: compute diffusion loss
        if gt_actions is not None and return_loss:
            return self._compute_training_loss(global_cond, gt_actions, action_valid)
        
        # 3. Inference mode: iterative denoising
        actions = self._diffusion_inference(global_cond, batch_size, device)
        
        # 4. Post-process actions
        final_actions = get_action_from_diffusion_output(
            actions, self.action_stats, cumulative=cumulative
        )
        
        if return_loss:
            return {
                'actions': final_actions,
                'normalized_actions': actions,
                'loss': torch.tensor(0.0, device=device),
            }
        
        return final_actions
    
    def _compute_training_loss(
        self,
        global_cond: torch.Tensor,
        gt_actions: torch.Tensor,
        action_valid: Optional[torch.Tensor] = None,  # 🆕 添加 mask 参数
    ) -> Dict[str, torch.Tensor]:
        """
        Compute diffusion training loss with proper masking.
        
        Args:
            global_cond: (B, encoding_size) conditioning
            gt_actions: (B, pred_horizon, action_dim) ground truth actions
            action_valid: Optional (B,) mask indicating which samples have valid actions
            
        Returns:
            Dict with 'loss', 'actions', 'normalized_actions'
        """
        device = global_cond.device
        batch_size = global_cond.shape[0]
        
        # 🔍 [DIAG] 验证输入维度
        # gt_actions 应该是 [B, pred_horizon, action_dim]
        if gt_actions.dim() == 2:
            # 如果是 [B, action_dim]，自动添加 pred_horizon 维度
            gt_actions = gt_actions.unsqueeze(1)
            logger.debug(f"[DIAG] gt_actions reshaped from 2D to 3D: {gt_actions.shape}")
        
        expected_shape = (batch_size, self.pred_horizon, self.action_dim)
        if gt_actions.shape != expected_shape:
            logger.warning(
                f"[DIAG] gt_actions shape mismatch: got {gt_actions.shape}, "
                f"expected {expected_shape}"
            )
        
        # Normalize ground truth actions
        from .utils import normalize_actions
        normalized_gt = normalize_actions(gt_actions, self.action_stats)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=device, dtype=torch.long
        )
        
        # Sample noise
        noise = torch.randn_like(normalized_gt)
        
        # Add noise to actions (forward diffusion)
        noisy_actions = self.noise_scheduler.add_noise(normalized_gt, noise, timesteps)
        
        # Predict noise
        noise_pred = self.noise_pred_net(
            sample=noisy_actions,
            timestep=timesteps,
            global_cond=global_cond,
        )
        
        # 🔧 [FIX] 加权 MSE Loss：非零动作权重更高
        # 问题：数据集中95.5%的转向动作是0，模型学会"输出0"就能获得极低Loss
        # 解决：增加非零动作的权重，强制模型学习转向
        
        # 计算动作的绝对值作为权重（非零动作权重更高）
        # normalized_gt: [B, pred_horizon, action_dim]
        action_magnitude = normalized_gt.abs()  # [B, H, D]
        # 权重：1 + 9 * |action|，范围 [1, 10]（因为归一化后action在[-1,1]）
        weight = 1.0 + 9.0 * action_magnitude.clamp(0, 1)
        
        # 加权 MSE
        squared_error = (noise_pred - noise).pow(2)
        weighted_error = weight * squared_error
        
        # 计算逐样本的加权 MSE loss
        per_sample_loss = weighted_error.mean(dim=(1, 2))  # [B]
        
        if action_valid is not None:
            # 只对有效样本计算 loss
            mask = action_valid.float().to(device)
            if mask.sum() > 0:
                diffusion_loss = (per_sample_loss * mask).sum() / mask.sum()
            else:
                # 没有有效样本，返回 0 loss
                diffusion_loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            # 无 mask，使用全部样本
            diffusion_loss = per_sample_loss.mean()
        
        # 🔧 [FIX-2] 方差约束：确保模型输出有变化，不能全是同一个值
        # 每10步计算一次
        self._training_step_counter += 1
        variance_loss = torch.tensor(0.0, device=device)
        final_actions = None
        pred_actions = None
        
        compute_variance_loss = (self._training_step_counter % 10 == 0)
        
        if compute_variance_loss or self._inference_interval == 0 or self._training_step_counter % self._inference_interval == 0:
            with torch.no_grad():
                pred_actions = self._diffusion_inference(global_cond, batch_size, device)
                final_actions = get_action_from_diffusion_output(
                    pred_actions, self.action_stats, cumulative=True
                )
            
            if compute_variance_loss and pred_actions is not None:
                # 方差约束：预测动作的标准差必须 >= 0.1
                # 如果 batch 内所有预测都一样（全零），则产生惩罚
                pred_std = pred_actions.std()
                variance_loss = nn.functional.relu(0.1 - pred_std)
        
        # 总损失 = 扩散损失 + 方差约束
        loss = diffusion_loss + 0.3 * variance_loss
        
        return {
            'loss': loss,
            'diffusion_loss': diffusion_loss,
            'variance_loss': variance_loss,
            'actions': final_actions,
            'normalized_actions': pred_actions,
            'noise_std': noise.std(),
            'noise_pred_std': noise_pred.std(),
            'mse': squared_error.mean(),
        }
    
    def compute_loss(
        self,
        llm_features: torch.Tensor,
        gt_actions: torch.Tensor,
        action_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        供外部调用的 loss 计算方法（不做推理监控，更快）。
        
        Args:
            llm_features: (B, cond_dim) 或 (B, seq_len, cond_dim) LLM 输出特征
            gt_actions: (B, pred_horizon, action_dim) ground truth actions
            action_valid: Optional (B,) mask indicating which samples have valid actions
            
        Returns:
            Dict with 'loss' and diagnostic info
        """
        # 先通过 condition_projector 投影到 encoding_size
        global_cond = self.condition_projector(llm_features)  # (B, encoding_size)
        
        device = global_cond.device
        batch_size = global_cond.shape[0]
        
        # Validate and reshape gt_actions if needed
        if gt_actions.dim() == 2:
            gt_actions = gt_actions.unsqueeze(1)
        
        expected_shape = (batch_size, self.pred_horizon, self.action_dim)
        if gt_actions.shape != expected_shape:
            logger.warning(
                f"gt_actions shape mismatch: got {gt_actions.shape}, expected {expected_shape}"
            )
        
        # Normalize ground truth actions
        from .utils import normalize_actions
        normalized_gt = normalize_actions(gt_actions, self.action_stats)
        
        # Sample random timesteps
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=device, dtype=torch.long
        )
        
        # Sample noise
        noise = torch.randn_like(normalized_gt)
        
        # Add noise to actions (forward diffusion)
        noisy_actions = self.noise_scheduler.add_noise(normalized_gt, noise, timesteps)
        
        # Predict noise
        noise_pred = self.noise_pred_net(
            sample=noisy_actions,
            timestep=timesteps,
            global_cond=global_cond,
        )
        
        # 🔧 [FIX] 加权 MSE Loss：非零动作权重更高
        # 问题：数据集中95.5%的转向动作是0，模型学会"输出0"就能获得极低Loss
        # 解决：增加非零动作的权重，强制模型学习转向
        
        # 计算动作的绝对值作为权重（非零动作权重更高）
        action_magnitude = normalized_gt.abs()  # [B, H, D]
        # 权重：1 + 9 * |action|，范围 [1, 10]
        weight = 1.0 + 9.0 * action_magnitude.clamp(0, 1)
        
        # 加权 MSE
        squared_error = (noise_pred - noise).pow(2)
        weighted_error = weight * squared_error
        
        # 计算逐样本的加权 MSE loss
        per_sample_loss = weighted_error.mean(dim=(1, 2))  # [B]
        
        # Apply action_valid mask
        if action_valid is not None:
            mask = action_valid.float().to(device)
            if mask.sum() > 0:
                loss = (per_sample_loss * mask).sum() / mask.sum()
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            loss = per_sample_loss.mean()
        
        # Return loss with diagnostic info
        with torch.no_grad():
            noise_std = noise.std()
            noise_pred_std = noise_pred.std()
            mse = squared_error.mean()
        
        return {
            'loss': loss,
            'noise_std': noise_std,
            'noise_pred_std': noise_pred_std,
            'mse': mse,
        }
    
    def _diffusion_inference(
        self,
        global_cond: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Iterative denoising to generate actions.
        
        Args:
            global_cond: (B, encoding_size) conditioning
            batch_size: Batch size
            device: Target device
            
        Returns:
            (B, pred_horizon, action_dim) normalized actions in [-1, 1]
        """
        # Initialize with random noise
        noisy_action = torch.randn(
            (batch_size, self.pred_horizon, self.action_dim),
            device=device,
            dtype=global_cond.dtype,
        )
        
        # Set scheduler timesteps
        self.noise_scheduler.set_timesteps(self.config.num_diffusion_iters)
        
        # Iterative denoising
        for t in self.noise_scheduler.timesteps:
            # Predict noise
            noise_pred = self.noise_pred_net(
                sample=noisy_action,
                timestep=t.unsqueeze(-1).repeat(batch_size).to(device),
                global_cond=global_cond,
            )
            
            # Remove noise (reverse diffusion step)
            noisy_action = self.noise_scheduler.step(
                model_output=noise_pred,
                timestep=t,
                sample=noisy_action,
            ).prev_sample
        
        return noisy_action
    
    def get_action(
        self,
        llm_features: torch.Tensor,
        cumulative: bool = True,
    ) -> torch.Tensor:
        """
        Convenience method for inference.
        
        Args:
            llm_features: (B, seq_len, cond_dim) LLM output features
            cumulative: If True, return cumulative positions
            
        Returns:
            (B, pred_horizon, action_dim) predicted actions
        """
        self.eval()
        with torch.no_grad():
            return self.forward(llm_features, cumulative=cumulative)
    
    def update_action_stats(self, stats: ActionStats):
        """Update action statistics for normalization."""
        self.action_stats = stats
        self.config.action_stats_min = stats.min
        self.config.action_stats_max = stats.max
        logger.info(f"Action stats updated: min={stats.min}, max={stats.max}")


def create_diffusion_action_head(
    cond_dim: int = 2048,
    action_dim: int = 2,
    pred_horizon: int = 1,
    device: str = "cuda",
    **kwargs
) -> DiffusionActionHead:
    """
    Factory function to create a DiffusionActionHead.
    
    Args:
        cond_dim: LLM feature dimension
        action_dim: Action output dimension
        pred_horizon: Number of action steps to predict
        device: Target device
        **kwargs: Additional config parameters
        
    Returns:
        Initialized DiffusionActionHead
    """
    config = DiffusionActionConfig(
        cond_dim=cond_dim,
        action_dim=action_dim,
        pred_horizon=pred_horizon,
        device=device,
        **kwargs
    )
    
    model = DiffusionActionHead(config)
    model = model.to(device)
    
    return model


"""
Transformer-based Action Head for VLN Navigation

参考 InternNav/internnav/model/basemodel/internvla_n1/navdp.py 实现
使用 Transformer Decoder + Diffusion Policy 生成导航轨迹

Architecture:
    LLM Features (B, seq_len, vlm_token_dim) 
        -> VLM Embed MLP (B, 1, token_dim)
        -> Transformer Decoder (noise prediction)
        -> DDPM Denoising Loop
        -> Trajectory (B, pred_horizon, action_dim)
"""

import math
import logging
from typing import Optional, Dict, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffusers.schedulers.scheduling_ddpm import DDPMScheduler

logger = logging.getLogger(__name__)


class SinusoidalPosEmb(nn.Module):
    """Sinusoidal positional embedding for diffusion timesteps"""
    
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class TransformerActionHead(nn.Module):
    """
    Transformer Decoder based Action Head with Diffusion Policy.
    
    参考 InternNav 的 NavDP_Policy_DPT_CriticSum_DAT 实现，
    但简化为只使用 VLM 特征作为条件（不使用 RGBD encoder）。
    
    Args:
        vlm_token_dim: VLM (Qwen3-VL) 输出的 token 维度
        token_dim: 内部 token 维度
        predict_size: 预测的轨迹步数
        temporal_depth: Transformer Decoder 层数
        heads: 注意力头数
        dropout: Dropout 比例
        action_dim: 动作维度 (3 for x, y, theta)
        num_train_timesteps: 扩散训练步数
        action_scale: 动作放大倍数
    """
    
    def __init__(
        self,
        vlm_token_dim: int = 1024,
        token_dim: int = 384,
        predict_size: int = 24,
        temporal_depth: int = 16,
        heads: int = 8,
        dropout: float = 0.1,
        action_dim: int = 3,
        num_train_timesteps: int = 20,
        action_scale: float = 4.0,
        input_dtype: str = "bf16",
    ):
        super().__init__()
        
        self.predict_size = predict_size
        self.token_dim = token_dim
        self.vlm_token_dim = vlm_token_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.temporal_depth = temporal_depth
        
        if input_dtype == "bf16":
            self.input_dtype = torch.bfloat16
        else:
            self.input_dtype = torch.float32
        
        # VLM token 投影 (参考 InternNav)
        self.vlm_embed_mlp = nn.Sequential(
            nn.Linear(vlm_token_dim, vlm_token_dim // 4),
            nn.ReLU(),
            nn.Linear(vlm_token_dim // 4, vlm_token_dim // 8),
            nn.ReLU(),
            nn.Linear(vlm_token_dim // 8, token_dim),
        )
        
        # 动作输入嵌入
        self.input_embed = nn.Linear(action_dim, token_dim)
        
        # 位置嵌入
        # cond_pos_embed: [time_embed, goal_embed] = 2 个 token
        self.cond_pos_embed = nn.Parameter(torch.zeros((1, 2, token_dim)))
        self.out_pos_embed = nn.Parameter(torch.zeros((1, predict_size, token_dim)))
        
        # Dropout
        self.drop = nn.Dropout(dropout)
        
        # 时间步嵌入
        self.time_emb = SinusoidalPosEmb(token_dim)
        
        # Transformer Decoder
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=token_dim,
            nhead=heads,
            dim_feedforward=4 * token_dim,
            dropout=dropout,
            activation='gelu',
            batch_first=True,
            norm_first=True,
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, 
            num_layers=temporal_depth
        )
        
        # 输出层
        self.layernorm = nn.LayerNorm(token_dim)
        self.action_head = nn.Linear(token_dim, action_dim)
        
        # Diffusion Scheduler
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon',
        )
        
        # 因果掩码 (下三角)
        self.register_buffer(
            'tgt_mask',
            self._generate_causal_mask(predict_size)
        )
        
        logger.info(
            f"TransformerActionHead initialized: "
            f"predict_size={predict_size}, token_dim={token_dim}, "
            f"temporal_depth={temporal_depth}, action_dim={action_dim}"
        )
    
    def _generate_causal_mask(self, size: int) -> torch.Tensor:
        """生成因果掩码（下三角）"""
        mask = torch.triu(torch.ones(size, size), diagonal=1)
        mask = mask.masked_fill(mask == 1, float('-inf'))
        return mask
    
    def sample_noise(self, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        采样噪声用于训练
        
        参考 InternNav navdp.py sample_noise()
        
        Args:
            action: (B, predict_size, action_dim) 真实轨迹
            
        Returns:
            noise: 采样的噪声
            time_embeds: 时间步嵌入
            noisy_action_embed: 加噪后的动作嵌入
        """
        noise = torch.randn(action.shape, dtype=action.dtype, device=action.device)
        
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (action.shape[0],), device=action.device
        ).long()
        
        time_embeds = self.time_emb(timesteps).unsqueeze(1)
        
        noisy_action = self.noise_scheduler.add_noise(action, noise, timesteps)
        noisy_action_embed = self.input_embed(noisy_action)
        
        return noise, time_embeds, noisy_action_embed
    
    def predict_noise(
        self, 
        last_actions: torch.Tensor, 
        timestep: torch.Tensor, 
        goal_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        预测噪声
        
        参考 InternNav navdp.py predict_noise()
        
        Args:
            last_actions: (B, predict_size, action_dim) 当前动作
            timestep: (1,) 当前时间步
            goal_embed: (B, 1, token_dim) VLM 条件嵌入
            
        Returns:
            noise_pred: (B, predict_size, action_dim) 预测的噪声
        """
        action_embeds = self.input_embed(last_actions)
        time_embeds = self.time_emb(timestep.to(last_actions.device)).unsqueeze(1)
        
        # 扩展 time_embeds 到 batch size
        if time_embeds.shape[0] == 1 and action_embeds.shape[0] > 1:
            time_embeds = time_embeds.expand(action_embeds.shape[0], -1, -1)
        
        # 条件嵌入: [time, goal]
        cond_embedding = torch.cat([time_embeds, goal_embed], dim=1)
        cond_embedding = cond_embedding + self.cond_pos_embed[:, :cond_embedding.size(1), :]
        
        # 动作嵌入 + 位置编码
        input_embedding = action_embeds + self.out_pos_embed[:, :self.predict_size, :]
        
        # Transformer Decoder
        output = self.decoder(
            tgt=input_embedding, 
            memory=cond_embedding, 
            tgt_mask=self.tgt_mask.to(input_embedding.device)
        )
        output = self.layernorm(output)
        output = self.action_head(output)
        
        return output
    
    def forward(
        self,
        llm_features: torch.Tensor,
        gt_trajectory: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """
        前向传播
        
        Args:
            llm_features: (B, seq_len, vlm_token_dim) 或 (B, vlm_token_dim) LLM 输出特征
            gt_trajectory: (B, predict_size, action_dim) 真实轨迹（已放大）
            return_loss: 是否返回 loss
            
        Returns:
            Dict with 'loss', 'noise_pred', 'trajectory' (推理时)
        """
        device = llm_features.device
        batch_size = llm_features.shape[0]
        
        # 1. 处理 LLM 特征
        if llm_features.dim() == 3:
            # (B, seq_len, dim) -> (B, dim) -> (B, 1, token_dim)
            llm_pooled = llm_features.mean(dim=1)
        else:
            llm_pooled = llm_features
        
        vlm_embed = self.vlm_embed_mlp(llm_pooled).unsqueeze(1)  # (B, 1, token_dim)
        
        # 2. 训练模式：计算 diffusion loss
        if gt_trajectory is not None and return_loss:
            return self._compute_training_loss(vlm_embed, gt_trajectory)
        
        # 3. 推理模式：迭代去噪
        trajectory = self._diffusion_inference(vlm_embed, batch_size, device)
        
        return {
            'trajectory': trajectory,
            'loss': torch.tensor(0.0, device=device),
        }
    
    def _compute_training_loss(
        self,
        vlm_embed: torch.Tensor,
        gt_trajectory: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算训练 loss
        
        参考 InternNav forward_vlm_traj()
        """
        device = vlm_embed.device
        
        # 采样噪声
        noise, time_embed, noisy_action_embed = self.sample_noise(gt_trajectory)
        
        # 条件嵌入
        cond_embed = torch.cat([time_embed, vlm_embed], dim=1)
        cond_embeddings = self.drop(cond_embed + self.cond_pos_embed[:, :cond_embed.size(1), :])
        
        # 动作嵌入
        action_embeddings = self.drop(noisy_action_embed + self.out_pos_embed[:, :self.predict_size, :])
        
        # Transformer Decoder
        output = self.decoder(
            tgt=action_embeddings, 
            memory=cond_embeddings, 
            tgt_mask=self.tgt_mask.to(device)
        )
        output = self.layernorm(output)
        noise_pred = self.action_head(output)
        
        # MSE Loss
        loss = F.mse_loss(noise_pred, noise)
        
        return {
            'loss': loss,
            'noise_pred': noise_pred,
            'noise': noise,
        }
    
    def _diffusion_inference(
        self,
        vlm_embed: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Diffusion 推理：迭代去噪生成轨迹
        
        参考 InternNav predict_pointgoal_action()
        """
        # 初始化随机噪声
        noisy_action = torch.randn(
            (batch_size, self.predict_size, self.action_dim),
            dtype=vlm_embed.dtype,
            device=device,
        )
        
        naction = noisy_action
        
        # 设置时间步
        self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
        
        # 迭代去噪
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.predict_noise(naction, k.unsqueeze(0), vlm_embed)
            naction = self.noise_scheduler.step(
                model_output=noise_pred, 
                timestep=k, 
                sample=naction
            ).prev_sample
        
        return naction
    
    def compute_loss(
        self,
        llm_features: torch.Tensor,
        gt_trajectory: torch.Tensor,
        trajectory_valid: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        供外部调用的 loss 计算方法
        
        Args:
            llm_features: (B, seq_len, vlm_token_dim) LLM 输出特征
            gt_trajectory: (B, predict_size, action_dim) 真实轨迹
            trajectory_valid: (B,) mask，指示哪些样本有效
            
        Returns:
            Dict with 'loss'
        """
        result = self.forward(llm_features, gt_trajectory, return_loss=True)
        
        # 应用 mask
        if trajectory_valid is not None:
            mask = trajectory_valid.float()
            if mask.sum() > 0:
                # 需要重新计算 per-sample loss
                noise_pred = result['noise_pred']
                noise = result['noise']
                per_sample_loss = F.mse_loss(noise_pred, noise, reduction='none').mean(dim=(1, 2))
                loss = (per_sample_loss * mask).sum() / mask.sum()
                result['loss'] = loss
        
        return result
    
    def get_trajectory(
        self,
        llm_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        推理接口：生成轨迹
        
        Args:
            llm_features: (B, seq_len, vlm_token_dim) LLM 输出特征
            
        Returns:
            (B, predict_size, action_dim) 预测轨迹
        """
        self.eval()
        with torch.no_grad():
            result = self.forward(llm_features)
            return result['trajectory']


def create_transformer_action_head(
    vlm_token_dim: int = 1024,
    token_dim: int = 384,
    predict_size: int = 24,
    temporal_depth: int = 16,
    heads: int = 8,
    dropout: float = 0.1,
    action_dim: int = 3,
    num_train_timesteps: int = 20,
    action_scale: float = 4.0,
    device: str = "cuda",
) -> TransformerActionHead:
    """Factory function to create TransformerActionHead"""
    
    model = TransformerActionHead(
        vlm_token_dim=vlm_token_dim,
        token_dim=token_dim,
        predict_size=predict_size,
        temporal_depth=temporal_depth,
        heads=heads,
        dropout=dropout,
        action_dim=action_dim,
        num_train_timesteps=num_train_timesteps,
        action_scale=action_scale,
    )
    
    model = model.to(device)
    
    return model

"""
Transformer-based Action Head for VLN Navigation

参考 InternNav/internnav/model/basemodel/diffusion_policy_modified/transformer_for_diffusion_modified.py
使用 Transformer Decoder + Diffusion Policy 生成导航轨迹

对齐 InternNav TransformerForDiffusion 的关键设计：
1. 完整的权重初始化 (_init_weights)
2. 条件编码器 (TransformerEncoder 预处理条件)
3. 简化 VLM 嵌入投影 (单层线性)
4. 因果掩码 (仅 tgt_mask，对齐 InternNav 不使用 memory_mask)
5. 位置嵌入正态初始化
6. Weight decay 参数分组 (get_optim_groups)

Architecture:
    LLM Features (B, seq_len, vlm_token_dim) 
        -> Condition Encoder (TransformerEncoder)
        -> Transformer Decoder (noise prediction)
        -> DDPM Denoising Loop
        -> Trajectory (B, pred_horizon, action_dim)
"""

import math
import logging
from typing import Optional, Dict, List, Tuple, Union
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
    
    对齐 InternNav 的 TransformerForDiffusion 实现：
    - 完整的权重初始化 (_init_weights)
    - 条件编码器 (TransformerEncoder)
    - 因果掩码 (仅 tgt_mask，对齐 InternNav 不使用 memory_mask)
    - Weight decay 参数分组 (get_optim_groups)
    
    Args:
        vlm_token_dim: VLM (Qwen3-VL) 输出的 token 维度
        n_emb: 内部嵌入维度 (对应 InternNav 的 n_emb)
        predict_size: 预测的轨迹步数 (horizon)
        n_layer: Transformer Decoder 层数
        n_head: 注意力头数 (默认 6，对齐 InternNav head_dim=64)
        n_cond_layers: 条件编码器层数 (0 表示不使用)
        p_drop_emb: Embedding dropout
        p_drop_attn: Attention dropout
        action_dim: 动作维度 (3 for x, y, theta)
        num_train_timesteps: 扩散训练步数
        causal_attn: 是否使用因果注意力
    """
    
    def __init__(
        self,
        vlm_token_dim: int = 1024,
        n_emb: int = 384,
        predict_size: int = 24,
        n_layer: int = 16,
        n_head: int = 6,  # 对齐 InternNav: n_emb // head_dim = 384 // 64 = 6
        n_cond_layers: int = 4,  # 条件编码器层数
        p_drop_emb: float = 0.1,
        p_drop_attn: float = 0.1,
        action_dim: int = 3,
        num_train_timesteps: int = 20,
        causal_attn: bool = True,
        n_obs_steps: int = 1,  # 观测步数（VLM pooled = 1）
        input_dtype: str = "bf16",
    ):
        super().__init__()
        
        self.predict_size = predict_size
        self.n_emb = n_emb
        self.vlm_token_dim = vlm_token_dim
        self.action_dim = action_dim
        self.n_layer = n_layer
        self.n_head = n_head
        self.n_obs_steps = n_obs_steps
        self.causal_attn = causal_attn
        self.use_encoder = n_cond_layers > 0
        
        if input_dtype == "bf16":
            self.input_dtype = torch.bfloat16
        else:
            self.input_dtype = torch.float32
        
        # ==================== 计算 token 数量 ====================
        T = predict_size  # 预测 horizon
        T_cond = 1 + n_obs_steps  # time + obs
        
        self.T = T
        self.T_cond = T_cond
        
        # ==================== Input Embedding ====================
        # 动作输入嵌入（参考 InternNav self.input_emb）
        self.input_emb = nn.Linear(action_dim, n_emb)
        
        # 位置嵌入（后面会用正态分布初始化）
        self.pos_emb = nn.Parameter(torch.zeros(1, T, n_emb))
        
        # Dropout
        self.drop = nn.Dropout(p_drop_emb)
        
        # ==================== Condition Encoder ====================
        # 时间步嵌入
        self.time_emb = SinusoidalPosEmb(n_emb)
        
        # VLM 条件嵌入（简化为单层线性，参考 InternNav self.cond_obs_emb）
        self.cond_obs_emb = nn.Linear(vlm_token_dim, n_emb)
        
        # 条件位置嵌入
        self.cond_pos_emb = nn.Parameter(torch.zeros(1, T_cond, n_emb))
        
        # 条件编码器（TransformerEncoder，参考 InternNav）
        self.encoder = None
        if self.use_encoder and n_cond_layers > 0:
            encoder_layer = nn.TransformerEncoderLayer(
                d_model=n_emb,
                nhead=n_head,
                dim_feedforward=4 * n_emb,
                dropout=p_drop_attn,
                activation='gelu',
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(
                encoder_layer=encoder_layer, 
                num_layers=n_cond_layers
            )
        
        # ==================== Transformer Decoder ====================
        decoder_layer = nn.TransformerDecoderLayer(
            d_model=n_emb,
            nhead=n_head,
            dim_feedforward=4 * n_emb,
            dropout=p_drop_attn,
            activation='gelu',
            batch_first=True,
            norm_first=True,  # 重要：稳定性
        )
        self.decoder = nn.TransformerDecoder(
            decoder_layer=decoder_layer, 
            num_layers=n_layer
        )
        
        # ==================== Output Head ====================
        self.ln_f = nn.LayerNorm(n_emb)
        self.head = nn.Linear(n_emb, action_dim)
        
        # ==================== Diffusion Scheduler ====================
        self.noise_scheduler = DDPMScheduler(
            num_train_timesteps=num_train_timesteps,
            beta_schedule='squaredcos_cap_v2',
            clip_sample=True,
            prediction_type='epsilon',
        )
        
        # ==================== Attention Masks ====================
        # 对齐 InternNav: 仅使用 tgt_mask (decoder self-attention causal mask)
        # 注意：InternNav 虽然注册了 memory_mask 但在 forward 中不使用它
        # InternNav 使用 memory_key_padding_mask (cond_mask) 用于 classifier-free guidance
        if causal_attn:
            sz = T
            mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
            mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
            self.register_buffer('mask', mask)  # 对齐 InternNav 命名: self.mask
        else:
            self.mask = None
        
        # ==================== 权重初始化 ====================
        self.apply(self._init_weights)
        
        logger.info(
            f"TransformerActionHead initialized: "
            f"predict_size={predict_size}, n_emb={n_emb}, "
            f"n_layer={n_layer}, n_cond_layers={n_cond_layers}, "
            f"action_dim={action_dim}, causal_attn={causal_attn}"
        )
        logger.info(f"  Total parameters: {sum(p.numel() for p in self.parameters()):,}")
    
    def _init_weights(self, module):
        """
        权重初始化（对齐 InternNav TransformerForDiffusion._init_weights）
        """
        ignore_types = (
            nn.Dropout,
            SinusoidalPosEmb,
            nn.TransformerEncoderLayer,
            nn.TransformerDecoderLayer,
            nn.TransformerEncoder,
            nn.TransformerDecoder,
            nn.ModuleList,
            nn.Sequential,
        )
        
        if isinstance(module, (nn.Linear, nn.Embedding)):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.MultiheadAttention):
            # 初始化注意力权重
            weight_names = ['in_proj_weight', 'q_proj_weight', 'k_proj_weight', 'v_proj_weight']
            for name in weight_names:
                weight = getattr(module, name)
                if weight is not None:
                    torch.nn.init.normal_(weight, mean=0.0, std=0.02)
            
            bias_names = ['in_proj_bias', 'bias_k', 'bias_v']
            for name in bias_names:
                bias = getattr(module, name)
                if bias is not None:
                    torch.nn.init.zeros_(bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, TransformerActionHead):
            # 位置嵌入使用正态分布初始化
            torch.nn.init.normal_(module.pos_emb, mean=0.0, std=0.02)
            if module.cond_obs_emb is not None:
                torch.nn.init.normal_(module.cond_pos_emb, mean=0.0, std=0.02)
        elif isinstance(module, ignore_types):
            # no param
            pass
        else:
            # 对齐 InternNav: 对未知模块发出警告（开发阶段有助于发现遗漏）
            logger.warning(f'Unaccounted module in _init_weights: {type(module).__name__}')
    
    def forward_diffusion(
        self,
        sample: torch.Tensor,
        timestep: Union[torch.Tensor, float, int],
        cond: torch.Tensor,
        cond_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Diffusion 前向传播（预测噪声）
        
        对齐 InternNav TransformerForDiffusion.forward()
        
        Args:
            sample: (B, T, action_dim) 加噪的动作序列
            timestep: (B,) 或标量 时间步
            cond: (B, n_obs_steps, vlm_token_dim) 条件（VLM 特征）
            cond_mask: Optional (B, T_cond) 条件 padding mask
                       (用于 classifier-free guidance，对齐 InternNav)
            
        Returns:
            noise_pred: (B, T, action_dim) 预测的噪声
        """
        device = sample.device
        batch_size = sample.shape[0]
        
        # 1. 处理时间步（对齐 InternNav）
        timesteps = timestep
        if not torch.is_tensor(timesteps):
            timesteps = torch.tensor([timesteps], dtype=torch.long, device=device)
        elif torch.is_tensor(timesteps) and len(timesteps.shape) == 0:
            timesteps = timesteps[None].to(device)
        # broadcast to batch dimension in a way that's compatible with ONNX/Core ML
        timesteps = timesteps.expand(sample.shape[0])
        time_emb = self.time_emb(timesteps).unsqueeze(1)  # (B, 1, n_emb)
        
        # 2. 条件编码器（对齐 InternNav encoder 路径）
        cond_embeddings = time_emb
        cond_obs_emb = self.cond_obs_emb(cond)  # (B, n_obs_steps, n_emb)
        cond_embeddings = torch.cat([cond_embeddings, cond_obs_emb], dim=1)  # (B, T_cond, n_emb)
        
        tc = cond_embeddings.shape[1]
        position_embeddings = self.cond_pos_emb[:, :tc, :]
        x = self.drop(cond_embeddings + position_embeddings)
        if self.use_encoder and self.encoder is not None:
            x = self.encoder(x)
        memory = x
        # (B, T_cond, n_emb)
        
        # 3. Decoder 输入
        token_embeddings = self.input_emb(sample)  # (B, T, n_emb)
        t = token_embeddings.shape[1]
        position_embeddings = self.pos_emb[:, :t, :]
        x = self.drop(token_embeddings + position_embeddings)
        # (B, T, n_emb)
        
        # 4. Transformer Decoder
        # 对齐 InternNav: 仅使用 tgt_mask，不使用 memory_mask
        # memory_key_padding_mask 用于 classifier-free guidance (可选)
        x = self.decoder(
            tgt=x,
            memory=memory,
            tgt_mask=self.mask,
            memory_key_padding_mask=cond_mask,
        )
        # (B, T, n_emb)
        
        # 5. Output head
        x = self.ln_f(x)
        x = self.head(x)
        # (B, T, action_dim)
        
        return x
    
    def get_optim_groups(self, weight_decay: float = 1e-3) -> List[Dict]:
        """
        参数分组（对齐 InternNav TransformerForDiffusion.get_optim_groups）
        
        将参数分为两组：
        - decay: Linear weights, MultiheadAttention weights
        - no_decay: biases, LayerNorm weights, Embedding weights, 位置嵌入
        
        Args:
            weight_decay: Weight decay value for decay group
            
        Returns:
            List of param group dicts for optimizer
        """
        decay = set()
        no_decay = set()
        whitelist_weight_modules = (
            torch.nn.Linear,
            torch.nn.MultiheadAttention,
        )
        blacklist_weight_modules = (torch.nn.LayerNorm, torch.nn.Embedding)
        
        for mn, m in self.named_modules():
            for pn, p in m.named_parameters():
                fpn = '%s.%s' % (mn, pn) if mn else pn  # full param name
                
                if pn.endswith('bias'):
                    no_decay.add(fpn)
                elif pn.startswith('bias'):
                    # MultiheadAttention bias starts with "bias"
                    no_decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, whitelist_weight_modules):
                    decay.add(fpn)
                elif pn.endswith('weight') and isinstance(m, blacklist_weight_modules):
                    no_decay.add(fpn)
        
        # 位置嵌入参数不做 weight decay
        no_decay.add('pos_emb')
        if self.cond_pos_emb is not None:
            no_decay.add('cond_pos_emb')
        
        # 验证所有参数都被分组
        param_dict = {pn: p for pn, p in self.named_parameters()}
        inter_params = decay & no_decay
        union_params = decay | no_decay
        
        if len(inter_params) > 0:
            logger.warning(f'Parameters in both decay/no_decay sets: {inter_params}')
        
        unaccounted = param_dict.keys() - union_params
        if len(unaccounted) > 0:
            logger.warning(f'Parameters not in decay/no_decay sets: {unaccounted}')
            # 未分类的参数默认不做 decay
            no_decay.update(unaccounted)
        
        optim_groups = [
            {
                'params': [param_dict[pn] for pn in sorted(list(decay)) if pn in param_dict],
                'weight_decay': weight_decay,
            },
            {
                'params': [param_dict[pn] for pn in sorted(list(no_decay)) if pn in param_dict],
                'weight_decay': 0.0,
            },
        ]
        return optim_groups
    
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
        
        # 1. 处理 LLM 特征 -> (B, n_obs_steps, vlm_token_dim)
        if llm_features.dim() == 2:
            # (B, dim) -> (B, 1, dim)
            cond = llm_features.unsqueeze(1)
        elif llm_features.dim() == 3:
            # (B, seq_len, dim) -> (B, 1, dim) 池化
            cond = llm_features.mean(dim=1, keepdim=True)
        else:
            cond = llm_features
        
        # 2. 训练模式：计算 diffusion loss
        if gt_trajectory is not None and return_loss:
            return self._compute_training_loss(cond, gt_trajectory)
        
        # 3. 推理模式：迭代去噪
        trajectory = self._diffusion_inference(cond, batch_size, device)
        
        return {
            'trajectory': trajectory,
            'loss': torch.tensor(0.0, device=device),
        }
    
    def _compute_training_loss(
        self,
        cond: torch.Tensor,
        gt_trajectory: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        计算训练 loss
        """
        device = cond.device
        batch_size = cond.shape[0]
        
        # 采样噪声
        noise = torch.randn_like(gt_trajectory)
        
        # 采样时间步
        timesteps = torch.randint(
            0, self.noise_scheduler.config.num_train_timesteps,
            (batch_size,), device=device
        ).long()
        
        # 加噪
        noisy_trajectory = self.noise_scheduler.add_noise(gt_trajectory, noise, timesteps)
        
        # 预测噪声
        noise_pred = self.forward_diffusion(noisy_trajectory, timesteps, cond)
        
        # MSE Loss
        loss = F.mse_loss(noise_pred, noise)
        
        return {
            'loss': loss,
            'noise_pred': noise_pred,
            'noise': noise,
        }
    
    def _diffusion_inference(
        self,
        cond: torch.Tensor,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        """
        Diffusion 推理：迭代去噪生成轨迹
        """
        # 初始化随机噪声
        noisy_trajectory = torch.randn(
            (batch_size, self.predict_size, self.action_dim),
            dtype=cond.dtype,
            device=device,
        )
        
        # 设置时间步
        self.noise_scheduler.set_timesteps(self.noise_scheduler.config.num_train_timesteps)
        
        # 迭代去噪
        for k in self.noise_scheduler.timesteps:
            noise_pred = self.forward_diffusion(noisy_trajectory, k, cond)
            noisy_trajectory = self.noise_scheduler.step(
                model_output=noise_pred, 
                timestep=k, 
                sample=noisy_trajectory
            ).prev_sample
        
        return noisy_trajectory
    
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
                # 重新计算 per-sample loss
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
    n_emb: int = 384,
    predict_size: int = 24,
    n_layer: int = 16,
    n_head: int = 6,
    n_cond_layers: int = 4,
    action_dim: int = 3,
    num_train_timesteps: int = 20,
    device: str = "cuda",
) -> TransformerActionHead:
    """Factory function to create TransformerActionHead"""
    
    model = TransformerActionHead(
        vlm_token_dim=vlm_token_dim,
        n_emb=n_emb,
        predict_size=predict_size,
        n_layer=n_layer,
        n_head=n_head,
        n_cond_layers=n_cond_layers,
        action_dim=action_dim,
        num_train_timesteps=num_train_timesteps,
    )
    
    model = model.to(device)
    
    return model

"""
Progress Prediction Head for VLN Navigation

替代二分类的 Stop 预测，使用连续值 (0-1) 表示任务完成进度。
当进度接近 1 时，表示应该停止。

对齐 InternNav 的 DistanceNetwork 实现：
- 使用 state + text_embed 拼接 (concat_state_txt 模式)
- 简单 3 层 MLP + ReLU + Sigmoid
- 简单 MSE loss

InternNav 参考代码 (rdp_policy.py:651-656):
```python
if self.model_config.progress_monitor.concat_state_txt:
    progress_pred = self.progress_monitor(
        torch.cat([state.squeeze(1), fused_update_txt_embeds[:, 0, :]], dim=1)
    )
```

Architecture:
    LLM Features (B, seq_len, D) -> [mean_pool, first_token] -> concat -> MLP -> Progress (0-1)
"""

import logging
from typing import Optional, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ProgressPredictionHead(nn.Module):
    """
    Progress Prediction Head (InternNav DistanceNetwork style with concat_state_txt).
    
    预测任务完成进度 (0-1)，替代二分类的 Stop 预测。
    
    对齐 InternNav 的关键设计：
    1. 使用 state + text_embed 拼接 (input_dim * 2)
    2. 3 层 MLP + ReLU + Sigmoid
    3. 简单 MSE loss
    
    在我们的架构中：
    - state = LLM 的 mean pooling (全局信息)
    - text_embed = LLM 的第一个 token (通常包含指令信息)
    
    Args:
        input_dim: Input dimension from LLM features (single feature)
        concat_state_txt: 是否使用 state + text 拼接模式 (default: True, 对齐 InternNav)
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,  # 保持接口兼容，但不使用
        token_dim: int = 384,   # 保持接口兼容，但不使用
        dropout: float = 0.1,   # 保持接口兼容，但不使用
        concat_state_txt: bool = True,  # 对齐 InternNav
    ):
        super().__init__()
        
        self.input_dim = input_dim
        self.concat_state_txt = concat_state_txt
        
        # 对齐 InternNav: 如果 concat_state_txt，input_dim 翻倍
        # state (mean pooling) + text_embed (first token) 拼接
        mlp_input_dim = input_dim * 2 if concat_state_txt else input_dim
        
        # 对齐 InternNav 的 DistanceNetwork 结构
        # 简单 3 层 MLP: input -> input/4 -> input/16 -> 1
        # 注意：这里的 input 是拼接后的维度
        self.progress_mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, mlp_input_dim // 4),
            nn.ReLU(),
            nn.Linear(mlp_input_dim // 4, mlp_input_dim // 16),
            nn.ReLU(),
            nn.Linear(mlp_input_dim // 16, 1),
            nn.Sigmoid(),  # 输出 0-1
        )
        
        # 初始化
        self._init_weights()
        
        logger.info(
            f"ProgressPredictionHead initialized (InternNav style): "
            f"input_dim={input_dim}, concat_state_txt={concat_state_txt}, "
            f"mlp_input_dim={mlp_input_dim}, "
            f"structure={mlp_input_dim}->{mlp_input_dim//4}->{mlp_input_dim//16}->1"
        )
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
    def _pool_features(self, llm_features: torch.Tensor) -> torch.Tensor:
        """
        Pool LLM features to fixed-size representation.
        
        对齐 InternNav 的 concat_state_txt 模式：
        - state = mean pooling (全局信息)
        - text_embed = first token (指令信息)
        
        Args:
            llm_features: (B, seq_len, D) or (B, D) LLM output features
            
        Returns:
            (B, D) or (B, 2*D) pooled features
        """
        if llm_features.dim() == 2:
            # 已经是 (B, D)，无法拼接
            return llm_features
        
        # (B, seq_len, D)
        if self.concat_state_txt:
            # 对齐 InternNav: state + text_embed 拼接
            state = llm_features.mean(dim=1)  # (B, D) - 全局信息
            text_embed = llm_features[:, 0, :]  # (B, D) - 第一个 token (指令)
            pooled = torch.cat([state, text_embed], dim=-1)  # (B, 2*D)
        else:
            # 简单 mean pooling
            pooled = llm_features.mean(dim=1)  # (B, D)
        
        return pooled
    
    def forward(
        self,
        llm_features: torch.Tensor,
        gt_progress: Optional[torch.Tensor] = None,
        action_valid: Optional[torch.Tensor] = None,
        return_loss: bool = False,
    ) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass for progress prediction.
        
        Args:
            llm_features: (B, seq_len, D) or (B, D) LLM output features
            gt_progress: Optional (B,) ground truth progress values (0-1)
            action_valid: Optional (B,) mask for valid actions
            return_loss: If True and gt_progress provided, return loss dict
            
        Returns:
            If return_loss and gt_progress provided:
                Dict with 'progress', 'loss'
            Else:
                (B,) progress values
        """
        # Pool features (对齐 InternNav concat_state_txt)
        pooled = self._pool_features(llm_features)
        
        # Predict progress: (B, 2*D) -> (B, 1) -> (B,)
        progress = self.progress_mlp(pooled).squeeze(-1)
        
        if gt_progress is not None and return_loss:
            loss = self._compute_loss(progress, gt_progress, action_valid)
            return {
                'progress': progress,
                'loss': loss,
            }
        
        return progress
    
    def compute_loss(
        self,
        progress: torch.Tensor,
        gt_progress: torch.Tensor,
        action_valid: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        供外部调用的 loss 计算方法。
        
        Args:
            progress: (B,) predicted progress values
            gt_progress: (B,) ground truth progress values
            action_valid: Optional (B,) mask for valid actions
            
        Returns:
            Scalar loss
        """
        return self._compute_loss(progress, gt_progress, action_valid)
    
    def _compute_loss(
        self,
        progress: torch.Tensor,
        targets: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute progress loss (InternNav style).
        
        使用简单 MSE loss，对齐 InternNav 实现。
        
        Args:
            progress: (B,) predicted progress
            targets: (B,) ground truth progress
            mask: Optional (B,) mask for valid samples
            
        Returns:
            Scalar loss
        """
        device = progress.device
        targets = targets.float().to(device)
        
        # 简单 MSE loss，对齐 InternNav
        if mask is not None:
            mask = mask.float().to(device)
            if mask.sum() > 0:
                mse_loss = F.mse_loss(progress, targets, reduction='none')
                loss = (mse_loss * mask).sum() / mask.sum()
            else:
                loss = torch.tensor(0.0, device=device, requires_grad=True)
        else:
            loss = F.mse_loss(progress, targets)
        
        return loss
    
    def predict_stop(
        self, 
        llm_features: torch.Tensor, 
        threshold: float = 0.9,
    ) -> torch.Tensor:
        """
        Predict stop action from progress.
        
        当 progress > threshold 时，预测为 STOP。
        
        Args:
            llm_features: (B, seq_len, D) or (B, D) LLM features
            threshold: Progress threshold for STOP prediction
            
        Returns:
            (B,) binary predictions (1=STOP, 0=continue)
        """
        with torch.no_grad():
            progress = self.forward(llm_features)
            return (progress > threshold).long()
    
    def get_progress(self, llm_features: torch.Tensor) -> torch.Tensor:
        """
        Get progress value for inference.
        
        Args:
            llm_features: (B, seq_len, D) or (B, D) LLM features
            
        Returns:
            (B,) progress values (0-1)
        """
        with torch.no_grad():
            return self.forward(llm_features)


def create_progress_head(
    input_dim: int = 1024,
    hidden_dim: int = 512,
    dropout: float = 0.1,
    concat_state_txt: bool = True,
    device: str = "cuda",
) -> ProgressPredictionHead:
    """Factory function to create ProgressPredictionHead"""
    
    model = ProgressPredictionHead(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
        concat_state_txt=concat_state_txt,
    )
    
    model = model.to(device)
    
    return model

"""
Progress Prediction Head for VLN Navigation

替代二分类的 Stop 预测，使用连续值 (0-1) 表示任务完成进度。
当进度接近 1 时，表示应该停止。

这种方法的优势：
1. 连续值更容易学习（MSE loss vs Binary CE）
2. 提供更丰富的监督信号（不只是 0/1）
3. 可以预测中间状态（如 "快到了" = 0.8）

对齐 InternNav 的 DistanceNetwork 实现：
- 简单 3 层 MLP + ReLU + Sigmoid
- 简单 MSE loss，无加权

Architecture:
    LLM Features (B, seq_len, D) -> Pooling -> MLP -> Progress (0-1)
"""

import logging
from typing import Optional, Dict, Union
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger(__name__)


class ProgressPredictionHead(nn.Module):
    """
    Progress Prediction Head (InternNav DistanceNetwork style).
    
    预测任务完成进度 (0-1)，替代二分类的 Stop 预测。
    
    进度定义：
        progress = 当前步数 / 总步数
    
    Args:
        input_dim: Input dimension from LLM features
        hidden_dim: Hidden layer dimension (unused, kept for compatibility)
        token_dim: Internal token dimension (unused, kept for compatibility)
        dropout: Dropout rate (unused, kept for compatibility)
    """
    
    def __init__(
        self,
        input_dim: int = 1024,
        hidden_dim: int = 512,  # 保持接口兼容，但不使用
        token_dim: int = 384,   # 保持接口兼容，但不使用
        dropout: float = 0.1,   # 保持接口兼容，但不使用
    ):
        super().__init__()
        
        self.input_dim = input_dim
        
        # 对齐 InternNav 的 DistanceNetwork 结构
        # 简单 3 层 MLP: input -> input/4 -> input/16 -> 1
        self.progress_mlp = nn.Sequential(
            nn.Linear(input_dim, input_dim // 4),
            nn.ReLU(),
            nn.Linear(input_dim // 4, input_dim // 16),
            nn.ReLU(),
            nn.Linear(input_dim // 16, 1),
            nn.Sigmoid(),  # 输出 0-1
        )
        
        # 初始化
        self._init_weights()
        
        logger.info(
            f"ProgressPredictionHead initialized (InternNav style): "
            f"input_dim={input_dim}, structure={input_dim}->{input_dim//4}->{input_dim//16}->1"
        )
    
    def _init_weights(self):
        """Initialize weights"""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
    
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
        # Pool if needed: (B, seq_len, D) -> (B, D)
        if llm_features.dim() == 3:
            pooled = llm_features.mean(dim=1)
        else:
            pooled = llm_features
        
        # Predict progress: (B, D) -> (B, 1) -> (B,)
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
        不使用边界加权，因为 progress 是均匀分布的。
        
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
    device: str = "cuda",
) -> ProgressPredictionHead:
    """Factory function to create ProgressPredictionHead"""
    
    model = ProgressPredictionHead(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        dropout=dropout,
    )
    
    model = model.to(device)
    
    return model

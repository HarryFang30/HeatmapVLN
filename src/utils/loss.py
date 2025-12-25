"""
Navigation Loss Functions for VLN Training.

包含：
1. NavigationHeatmapLoss: 标准热力图损失（适合平滑热力图）
2. HighFreqHeatmapLoss: 高频感知热力图损失（适合精确定位）
3. FocalHeatmapLoss: Focal + Dice 损失（适合稀疏热力图）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple


class NavigationHeatmapLoss(nn.Module):
    """
    标准导航热力图损失（原版）。
    
    适用场景：热力图是平滑的高斯分布
    
    包含：
    - Weighted MSE: 空间回归
    - KL Divergence: 分布对齐
    - BCE: 有效性分类
    """
    
    def __init__(self, alpha=20.0, lambda_mse=1.0, lambda_kl=0.2, lambda_valid=0.1):
        """
        Args:
            alpha: 像素加权系数。越高则模型越关注亮斑区域。
            lambda_mse: 空间回归 Loss 权重。
            lambda_kl: 分布对齐 Loss 权重。
            lambda_valid: 有效性分类 Loss 权重。
        """
        super().__init__()
        self.alpha = alpha
        self.lambda_mse = lambda_mse
        self.lambda_kl = lambda_kl
        self.lambda_valid = lambda_valid
        self.bce_loss = nn.BCEWithLogitsLoss()

    def _remove_ripples(self, gt_heatmap):
        """
        可选：平滑 GT 热力图（移除时间纹理）。
        注意：这会损失高频信息！
        """
        with torch.no_grad():
            gt_filled = F.max_pool2d(gt_heatmap, kernel_size=5, stride=1, padding=2)
            gt_smooth = F.avg_pool2d(gt_filled, kernel_size=5, stride=1, padding=2)
            B = gt_smooth.shape[0]
            max_vals = gt_smooth.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1) + 1e-6
            gt_final = gt_smooth / max_vals
        return gt_final

    def forward(self, pred_logits, gt_heatmap_raw, pred_validity=None, gt_validity=None, smooth_gt=True):
        """
        Args:
            pred_logits: [B, 1, H, W] 模型输出 (未经过 Sigmoid)
            gt_heatmap_raw: [B, 1, H, W] GT 热力图
            pred_validity: [B, 1] 有效性预测 (可选)
            gt_validity: [B, 1] 真实有效性 (可选)
            smooth_gt: 是否平滑 GT（默认 True，设为 False 保留高频）
        """
        # 1. GT 预处理
        if smooth_gt:
            target_map = self._remove_ripples(gt_heatmap_raw)
        else:
            # 保留原始高频信息，只归一化
            with torch.no_grad():
                B = gt_heatmap_raw.shape[0]
                max_vals = gt_heatmap_raw.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1) + 1e-6
                target_map = gt_heatmap_raw / max_vals

        # 2. Weighted MSE
        pred_sigmoid = torch.sigmoid(pred_logits)
        pixel_weights = 1.0 + self.alpha * target_map
        mse_loss = (pixel_weights * (pred_sigmoid - target_map) ** 2).mean()

        # 3. KL Divergence
        B = pred_logits.shape[0]
        flat_pred = pred_logits.view(B, -1)
        flat_target = target_map.view(B, -1)
        log_prob_pred = F.log_softmax(flat_pred, dim=1)
        target_sum = flat_target.sum(dim=1, keepdim=True) + 1e-6
        prob_target = flat_target / target_sum
        kl_loss = F.kl_div(log_prob_pred, prob_target, reduction='none').sum(dim=1)

        if gt_validity is not None:
            valid_mask = gt_validity.view(-1) > 0.5
            if valid_mask.sum() > 0:
                kl_loss = kl_loss[valid_mask].mean()
            else:
                kl_loss = torch.tensor(0.0, device=pred_logits.device)
        else:
            kl_loss = kl_loss.mean()

        # 4. Validity Loss
        if pred_validity is not None and gt_validity is not None:
            valid_loss = self.bce_loss(pred_validity, gt_validity)
        else:
            valid_loss = torch.tensor(0.0, device=pred_logits.device)

        # 总 Loss
        total_loss = (self.lambda_mse * mse_loss +
                      self.lambda_kl * kl_loss +
                      self.lambda_valid * valid_loss)

        return total_loss, {
            "mse": mse_loss.item(),
            "kl": kl_loss.item() if isinstance(kl_loss, torch.Tensor) else kl_loss,
            "valid": valid_loss.item() if isinstance(valid_loss, torch.Tensor) else valid_loss,
        }


class HighFreqHeatmapLoss(nn.Module):
    """
    高频感知热力图损失。
    
    适用场景：需要精确定位、边缘清晰的热力图
    
    特点：
    - 不平滑 GT
    - 添加 Laplacian (边缘) 损失
    - 添加 Focal 权重（聚焦难样本）
    """
    
    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_laplacian: float = 0.5,
        lambda_kl: float = 0.2,
        focal_gamma: float = 2.0,
        peak_weight: float = 10.0,
    ):
        """
        Args:
            lambda_mse: MSE 损失权重
            lambda_laplacian: 边缘/高频损失权重
            lambda_kl: 分布对齐损失权重
            focal_gamma: Focal 损失的 gamma 参数
            peak_weight: 峰值区域的额外权重
        """
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_laplacian = lambda_laplacian
        self.lambda_kl = lambda_kl
        self.focal_gamma = focal_gamma
        self.peak_weight = peak_weight
        
        # Laplacian kernel for edge detection
        laplacian_kernel = torch.tensor([
            [0, 1, 0],
            [1, -4, 1],
            [0, 1, 0]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        self.register_buffer('laplacian_kernel', laplacian_kernel)
    
    def _compute_laplacian(self, x: torch.Tensor) -> torch.Tensor:
        """计算 Laplacian（边缘/高频响应）"""
        # x: [B, 1, H, W]
        return F.conv2d(x, self.laplacian_kernel.to(x.device), padding=1)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred: [B, 1, H, W] 模型预测（已经 sigmoid 或在 [0,1] 范围）
            target: [B, 1, H, W] GT 热力图
            
        Returns:
            total_loss, loss_dict
        """
        # 确保在 [0, 1] 范围
        pred = pred.clamp(0, 1)
        
        # 归一化 target
        B = target.shape[0]
        max_vals = target.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1) + 1e-6
        target_norm = target / max_vals
        
        # 1. Focal-weighted MSE
        # 越接近峰值，权重越高；难样本（预测偏差大）权重也越高
        diff = (pred - target_norm).abs()
        focal_weight = (1 + diff) ** self.focal_gamma
        peak_weight = 1 + self.peak_weight * target_norm
        weighted_mse = (focal_weight * peak_weight * diff ** 2).mean()
        
        # 2. Laplacian (高频) 损失
        pred_laplacian = self._compute_laplacian(pred)
        target_laplacian = self._compute_laplacian(target_norm)
        laplacian_loss = F.l1_loss(pred_laplacian, target_laplacian)
        
        # 3. KL Divergence
        flat_pred = pred.view(B, -1)
        flat_target = target_norm.view(B, -1)
        log_prob_pred = F.log_softmax(flat_pred * 10, dim=1)  # Temperature
        target_sum = flat_target.sum(dim=1, keepdim=True) + 1e-6
        prob_target = flat_target / target_sum
        kl_loss = F.kl_div(log_prob_pred, prob_target, reduction='batchmean')
        
        # 总 Loss
        total_loss = (
            self.lambda_mse * weighted_mse +
            self.lambda_laplacian * laplacian_loss +
            self.lambda_kl * kl_loss
        )
        
        return total_loss, {
            "mse": weighted_mse.item(),
            "laplacian": laplacian_loss.item(),
            "kl": kl_loss.item(),
        }


class FocalHeatmapLoss(nn.Module):
    """
    Focal + Dice 热力图损失。
    
    适用场景：极度稀疏的热力图（只有很小的亮斑）
    
    特点：
    - Focal Loss: 解决类别不平衡
    - Dice Loss: 关注区域重叠
    """
    
    def __init__(
        self,
        alpha: float = 0.25,
        gamma: float = 2.0,
        lambda_focal: float = 1.0,
        lambda_dice: float = 1.0,
        smooth: float = 1e-6,
    ):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.lambda_focal = lambda_focal
        self.lambda_dice = lambda_dice
        self.smooth = smooth
    
    def _focal_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Binary Focal Loss"""
        bce = F.binary_cross_entropy(pred, target, reduction='none')
        pt = torch.where(target > 0.5, pred, 1 - pred)
        focal_weight = (1 - pt) ** self.gamma
        alpha_weight = torch.where(target > 0.5, self.alpha, 1 - self.alpha)
        return (alpha_weight * focal_weight * bce).mean()
    
    def _dice_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Dice Loss for soft labels"""
        pred_flat = pred.view(-1)
        target_flat = target.view(-1)
        intersection = (pred_flat * target_flat).sum()
        dice = (2 * intersection + self.smooth) / (
            pred_flat.sum() + target_flat.sum() + self.smooth
        )
        return 1 - dice
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred: [B, 1, H, W] 预测（sigmoid 后）
            target: [B, 1, H, W] GT
        """
        pred = pred.clamp(1e-6, 1 - 1e-6)
        
        # 归一化 target
        B = target.shape[0]
        max_vals = target.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1) + 1e-6
        target_norm = target / max_vals
        
        focal = self._focal_loss(pred, target_norm)
        dice = self._dice_loss(pred, target_norm)
        
        total_loss = self.lambda_focal * focal + self.lambda_dice * dice
        
        return total_loss, {
            "focal": focal.item(),
            "dice": dice.item(),
        }


class CombinedNavigationLoss(nn.Module):
    """
    组合损失：热力图 + 动作。
    
    用于训练时自动平衡两个任务。
    """
    
    def __init__(
        self,
        heatmap_loss: nn.Module,
        action_weight: float = 0.5,
        heatmap_weight: float = 1.0,
        use_uncertainty_weighting: bool = False,
    ):
        """
        Args:
            heatmap_loss: 热力图损失模块
            action_weight: 动作损失权重
            heatmap_weight: 热力图损失权重
            use_uncertainty_weighting: 是否使用 uncertainty weighting（自动平衡）
        """
        super().__init__()
        self.heatmap_loss = heatmap_loss
        self.action_weight = action_weight
        self.heatmap_weight = heatmap_weight
        self.use_uncertainty_weighting = use_uncertainty_weighting
        
        if use_uncertainty_weighting:
            # Learnable log(sigma^2) for uncertainty weighting
            self.log_var_heatmap = nn.Parameter(torch.zeros(1))
            self.log_var_action = nn.Parameter(torch.zeros(1))
    
    def forward(
        self,
        pred_heatmap: torch.Tensor,
        gt_heatmap: torch.Tensor,
        action_loss: torch.Tensor,
        pred_validity: Optional[torch.Tensor] = None,
        gt_validity: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred_heatmap: 预测热力图
            gt_heatmap: GT 热力图
            action_loss: 动作损失（来自 DiffusionActionHead）
            pred_validity: 有效性预测
            gt_validity: GT 有效性
        """
        # 热力图损失
        if pred_validity is not None and gt_validity is not None:
            hm_loss, hm_dict = self.heatmap_loss(
                pred_heatmap, gt_heatmap, pred_validity, gt_validity
            )
        else:
            hm_loss, hm_dict = self.heatmap_loss(pred_heatmap, gt_heatmap)
        
        if self.use_uncertainty_weighting:
            # Uncertainty weighting: L = L1/σ1² + L2/σ2² + log(σ1²) + log(σ2²)
            precision_hm = torch.exp(-self.log_var_heatmap)
            precision_act = torch.exp(-self.log_var_action)
            
            total_loss = (
                precision_hm * hm_loss + self.log_var_heatmap +
                precision_act * action_loss + self.log_var_action
            )
            
            hm_dict['log_var_hm'] = self.log_var_heatmap.item()
            hm_dict['log_var_act'] = self.log_var_action.item()
        else:
            total_loss = (
                self.heatmap_weight * hm_loss +
                self.action_weight * action_loss
            )
        
        hm_dict['action_loss'] = action_loss.item()
        hm_dict['total_loss'] = total_loss.item()
        
        return total_loss, hm_dict


# ============================================
# Diffusion-specific losses
# ============================================

class DiffusionHeatmapLoss(nn.Module):
    """
    专门用于 Diffusion Heatmap Head 的监督损失。
    
    与 MSE noise prediction 不同，这个直接监督最终输出。
    用于 warmup 阶段或混合训练。
    """
    
    def __init__(
        self,
        lambda_mse: float = 1.0,
        lambda_peak: float = 2.0,
        lambda_edge: float = 0.5,
    ):
        super().__init__()
        self.lambda_mse = lambda_mse
        self.lambda_peak = lambda_peak
        self.lambda_edge = lambda_edge
        
        # Sobel kernel for edge detection
        sobel_x = torch.tensor([
            [-1, 0, 1],
            [-2, 0, 2],
            [-1, 0, 1]
        ], dtype=torch.float32).view(1, 1, 3, 3)
        sobel_y = sobel_x.transpose(2, 3)
        self.register_buffer('sobel_x', sobel_x)
        self.register_buffer('sobel_y', sobel_y)
    
    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Args:
            pred: [B, H, W] 预测热力图
            target: [B, H, W] GT 热力图
        """
        # 确保维度正确
        if pred.dim() == 3:
            pred = pred.unsqueeze(1)
        if target.dim() == 3:
            target = target.unsqueeze(1)
        
        # 归一化 target
        B = target.shape[0]
        max_vals = target.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1) + 1e-6
        target_norm = target / max_vals
        pred_norm = pred.clamp(0, 1)
        
        # 1. 基础 MSE
        mse_loss = F.mse_loss(pred_norm, target_norm)
        
        # 2. 峰值损失（强调高值区域的精度）
        peak_mask = (target_norm > 0.5).float()
        peak_loss = ((pred_norm - target_norm) ** 2 * peak_mask).sum() / (peak_mask.sum() + 1)
        
        # 3. 边缘损失（高频）
        pred_edge_x = F.conv2d(pred_norm, self.sobel_x.to(pred.device), padding=1)
        pred_edge_y = F.conv2d(pred_norm, self.sobel_y.to(pred.device), padding=1)
        pred_edge = torch.sqrt(pred_edge_x ** 2 + pred_edge_y ** 2 + 1e-6)
        
        target_edge_x = F.conv2d(target_norm, self.sobel_x.to(target.device), padding=1)
        target_edge_y = F.conv2d(target_norm, self.sobel_y.to(target.device), padding=1)
        target_edge = torch.sqrt(target_edge_x ** 2 + target_edge_y ** 2 + 1e-6)
        
        edge_loss = F.l1_loss(pred_edge, target_edge)
        
        total_loss = (
            self.lambda_mse * mse_loss +
            self.lambda_peak * peak_loss +
            self.lambda_edge * edge_loss
        )
        
        return total_loss, {
            "mse": mse_loss.item(),
            "peak": peak_loss.item(),
            "edge": edge_loss.item(),
        }

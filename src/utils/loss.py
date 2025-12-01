import torch
import torch.nn as nn
import torch.nn.functional as F

class NavigationHeatmapLoss(nn.Module):
    def __init__(self, alpha=20.0, lambda_mse=1.0, lambda_kl=0.2, lambda_valid=0.1):
        """
        Args:
            alpha: 像素加权系数。越高则模型越关注亮斑区域（解决稀疏性）。
            lambda_mse: 空间回归 Loss 权重。
            lambda_kl: 分布对齐 Loss 权重。
            lambda_valid: 有效性分类 Loss 权重。
        """
        super().__init__()
        self.alpha = alpha
        self.lambda_mse = lambda_mse
        self.lambda_kl = lambda_kl
        self.lambda_valid = lambda_valid

        # 用于存在性预测的二分类 Loss
        self.bce_loss = nn.BCEWithLogitsLoss()

    def _remove_ripples(self, gt_heatmap):
        """
        关键步骤：在线处理 Ground Truth。
        将带有时间纹理（波纹）的热力图转换为纯净的空间概率分布图。

        原理：
        1. MaxPool (Dilation): 填补波纹中间的黑色空隙（波谷）。
        2. AvgPool (Blur): 平滑边缘，恢复成类高斯分布。
        """
        with torch.no_grad():
            # 1. 膨胀操作：填满同心圆之间的缝隙
            # kernel_size 需要略大于你的波纹波长，5或7通常合适
            gt_filled = F.max_pool2d(gt_heatmap, kernel_size=5, stride=1, padding=2)

            # 2. 平滑操作：让分布更自然
            gt_smooth = F.avg_pool2d(gt_filled, kernel_size=5, stride=1, padding=2)

            # 3. 重新归一化：确保最大值仍为 1.0
            B = gt_smooth.shape[0]
            max_vals = gt_smooth.view(B, -1).max(dim=1)[0].view(B, 1, 1, 1) + 1e-6
            gt_final = gt_smooth / max_vals

        return gt_final

    def forward(self, pred_logits, gt_heatmap_raw, pred_validity, gt_validity):
        """
        Args:
            pred_logits: [B, 1, H, W] 模型输出 (未经过 Sigmoid)
            gt_heatmap_raw: [B, 1, H, W] 采集脚本生成的原始 GT (带波纹)
            pred_validity: [B, 1] 模型预测该帧是否有效 (Logits)
            gt_validity: [B, 1] 真实有效性 (0/1)
        """

        # ==========================================
        # 1. GT 预处理：移除时间纹理，只留空间信息
        # ==========================================
        target_map = self._remove_ripples(gt_heatmap_raw)

        # ==========================================
        # 2. Loss A: Weighted MSE (精准定位)
        # ==========================================
        pred_sigmoid = torch.sigmoid(pred_logits)

        # 动态权重矩阵：让模型重点关注有值的区域
        # 目标值越高，权重越大；背景区域权重为 1
        pixel_weights = 1.0 + self.alpha * target_map

        mse_loss = (pixel_weights * (pred_sigmoid - target_map) ** 2).mean()

        # ==========================================
        # 3. Loss B: KL Divergence (全局分布对齐)
        # ==========================================
        # 将热力图视为概率分布，强迫模型学到"整体方向"
        # 仅对有效的样本计算 KL

        B = pred_logits.shape[0]
        flat_pred = pred_logits.view(B, -1)
        flat_target = target_map.view(B, -1)

        # 预测值转 Log Softmax
        log_prob_pred = F.log_softmax(flat_pred, dim=1)

        # 目标值转概率分布 (归一化到和为1)
        target_sum = flat_target.sum(dim=1, keepdim=True) + 1e-6
        prob_target = flat_target / target_sum

        kl_loss = F.kl_div(log_prob_pred, prob_target, reduction='none').sum(dim=1)

        # Mask掉无效的帧 (全黑图不能算KL)
        valid_mask = gt_validity.view(-1) > 0.5
        if valid_mask.sum() > 0:
            kl_loss = kl_loss[valid_mask].mean()
        else:
            kl_loss = torch.tensor(0.0, device=pred_logits.device)

        # ==========================================
        # 4. Loss C: Validity (有没有路?)
        # ==========================================
        valid_loss = self.bce_loss(pred_validity, gt_validity)

        # ==========================================
        # 总 Loss
        # ==========================================
        total_loss = (self.lambda_mse * mse_loss +
                      self.lambda_kl * kl_loss +
                      self.lambda_valid * valid_loss)

        return total_loss, {
            "mse": mse_loss.item(),
            "kl": kl_loss.item(),
            "valid": valid_loss.item()
        }

# VLN 任务 Loss 设计文档

## 1. 任务概述

本项目是一个 **视觉语言导航 (VLN)** 系统，基于冻结的 Qwen3-VL-8B 视觉语言模型，训练三个任务头：

| 任务头 | 输出 | 目的 |
|--------|------|------|
| **热力图生成头** | 64×64 概率分布图 | 预测当前位置在历史轨迹中的可能位置 |
| **轨迹预测头** | 24步 × 3维 (dx, dy, yaw) | 预测未来24步的运动轨迹 |
| **进度预测头** | 1维标量 [0, 1] | 预测当前在整体导航任务中的进度 |

---

## 2. 热力图生成 Loss

### 2.1 任务特点

- **输入**: Qwen3-VL 提取的视觉语言特征 (1024维) + 观察图像
- **输出**: 64×64 热力图，表示位置概率分布
- **模型**: DiffusionHeatmapHead (ConditionalUnet2D + DDPM)
- **GT特点**:
  - 高斯分布形状（可能多峰，取决于导航场景）
  - ~93.5% 是黑色背景 (值接近0)
  - ~6.5% 是峰值区域 (中心最大值~1.0)
- **特殊处理**: 360° 全景图，左右边界连续 (Circular Padding)

### 2.2 样本级权重设计

```python
# 检测正/负样本
is_negative = (gt_heatmap.max() < 0.01).float()  # 负样本：GT全零（不可见）
is_positive = 1.0 - is_negative

# 样本级权重：正样本提升 + 负样本降权
sample_weight = (
    positive_sample_boost * is_positive      # 正样本: 3.0x
    + negative_sample_weight * is_negative   # 负样本: 0.3x
)
```

| 样本类型 | 权重 | 理由 |
|----------|------|------|
| 正样本 (有峰值) | 3.0 | 峰值区域是导航的关键信息，需要重点学习 |
| 负样本 (全零) | 0.3 | 负样本信号稀疏，过度学习会导致热力图全黑 |

### 2.3 空间 Focal 权重

```python
# 所有像素共用的空间 focal 权重
focal_alpha = 20.0
spatial_focal = 1.0 + focal_alpha * gt_clamped  # 峰值处 15-21x
spatial_focal = spatial_focal / spatial_focal.mean()  # 归一化保持 loss 尺度
```

**核心思想**：让模型更关注峰值区域，峰值处 focal 权重高达 15-21x。

### 2.4 主损失：Epsilon Loss (Aggressive Focal)

```python
# 1. Epsilon Loss（噪声预测损失）
per_pixel_mse = (noise_pred - noise) ** 2

# 基础 MSE + 空间 focal + 样本权重
base_loss = (sample_weight_4d * per_pixel_mse).mean()
focal_loss = (sample_weight_4d * spatial_focal * per_pixel_mse).mean()

# Aggressive focal: 80% focal + 20% base
focal_weight = 0.8
diffusion_loss = (1 - focal_weight) * base_loss + focal_weight * focal_loss
```

**公式**:
$$L_{\epsilon} = \mathbb{E}_{t, \epsilon} \left[ ||\epsilon - \epsilon_\theta(x_t, t, c)||^2 \right]$$

**设计理由**：
- focal_weight 从 0.3 提升到 **0.8**，更 aggressive 地聚焦峰值区域
- 配合样本级权重 (正样本 3x)，形成双重 focal 机制

### 2.5 辅助损失

#### 2.5.1 x0 重构损失 (SNR 门控 + Focal)

```python
# 从噪声预测反推 x0 估计
x0_hat = (noisy_heatmap - sqrt(1-alpha_bar) * noise_pred) / sqrt(alpha_bar)

# SNR 软权重：低时间步(高 SNR)权重大，高时间步权重趋零
snr_weight = (snr / (snr + 1.0)).detach()  # sigmoid-like, [0,1]

# x0 loss + focal + SNR 门控 + 样本权重
x0_loss = (snr_weight * sample_weight_4d * spatial_focal * per_pixel_x0_mse).mean()
```

**设计理由**：
- x0_loss_weight 从 1.0 提升到 **3.0**，成为主力 loss
- SNR 门控：高时间步 x0_hat 接近噪声，梯度有害，动态跳过
- Focal 权重：峰值区域更关注

#### 2.5.2 Dice Loss (新增！专治稀疏信号)

```python
# 只在低噪声时计算（SNR > 1），否则 x0_hat 无意义
x0_pos = ((x0_hat + 1) / 2).clamp(0, 1)  # 预测
gt_pos = gt_heatmap.clamp(0, 1)            # GT

intersection = (p_flat * g_flat).sum(1)
union = p_flat.sum(1) + g_flat.sum(1)
dice = (2 * intersection + 1) / (union + 1)
dice_loss = (1 - dice).mean()
```

**设计理由**：
- 专治稀疏信号：Dice Loss 直接优化交集/并集，比 MSE 更适合稀疏分布
- SNR 门控：只对低噪声样本计算
- dice_loss_weight = **2.0**

#### 2.5.3 L1 稀疏性正则化

```python
# 鼓励输出大部分为 0
x0_denorm = (x0_hat + 1) / 2  # 还原到 [0, 1]
sparsity_loss = x0_denorm.mean()
```

**设计理由**：
- sparsity_loss_weight 从 0.5 降低到 **0.1**，避免过度稀疏化

#### 2.5.4 负样本零目标损失 (SNR 门控)

```python
# 对负样本（GT=全零），约束 x0 应为全 -1
neg_zero_loss = MSE(neg_x0, -1.0)
```

**设计理由**：
- SNR 门控：只对 SNR > 1 的样本施加
- 帮助模型学习"不可见区域应输出全零"

#### 2.5.5 多峰感知峰值距离损失 (SNR 门控)

```python
# 检测 GT 中的所有峰值，计算预测位置与 GT 的 L2 距离
# 1. NMS 检测 GT 峰值
# 2. 创建高斯注意力窗口
# 3. Soft-argmax 提取预测位置
# 4. 计算 L2 距离
peak_dist_loss = multi_peak_distance(snr_x0, snr_gt)
```

**设计理由**：
- peak_distance_loss_weight 从 2.0 提升到 **5.0**
- SNR 门控：只对低噪声样本计算
- 多峰感知：每个峰独立计算距离

### 2.6 可见性预测损失

```python
# 可见性预测头：判断当前视角是否能看到历史点
gt_has_peak = (gt_heatmap.max() > 0.01).float()
visibility_loss = F.binary_cross_entropy_with_logits(vis_logit, gt_has_peak)
```

**设计理由**：
- 显式建模"是否可见"，消除假阳性
- visibility_loss_weight = **0.5**

### 2.7 为什么使用 Diffusion？

对于可能出现 **多峰分布** 的热力图（1个峰、3个峰、7个峰都可能），Diffusion 是最佳选择：

| 方案 | 单峰 | 多峰 | 原因 |
|------|------|------|------|
| CNN + KL | ✅ | ❌ | 多峰时学到"糊状"输出 |
| CNN + 固定K点 | ❌ | ❌ | 峰数量不匹配 |
| **Diffusion** | ✅ | ✅ | 去噪过程自然"雕刻"任意数量的峰 |

### 2.8 完整 Loss 组合

```
Total Loss = diffusion_loss (80% focal)
           + x0_loss × 3.0
           + dice_loss × 2.0
           + sparsity_loss × 0.1
           + neg_zero_loss × 0.5
           + peak_dist_loss × 5.0
           + visibility_loss × 0.5
```

### 2.9 当前权重配置

```yaml
model:
  heatmap_head:
    use_circular_padding: true   # 360° 全景图支持
    positive_sample_boost: 3.0  # 正样本权重提升
    negative_sample_weight: 0.3 # 负样本降权

    x0_loss_weight: 3.0         # x0 重构损失权重 (1→3，主力)
    dice_loss_weight: 2.0       # Dice Loss (新增)
    sparsity_loss_weight: 0.1   # 稀疏性损失权重 (0.5→0.1)
    visibility_loss_weight: 0.5  # 可见性损失权重
    negative_sample_weight: 0.3  # 负样本降权
    peak_distance_loss_weight: 5.0 # 峰值距离损失权重 (2→5)

loss:
  history_weight: 1.0
  future_weight: 0.0   # 关闭未来热力图
```

---

## 3. 轨迹预测 Loss

### 3.1 任务特点

- **输入**: Qwen3-VL 特征 (1024维)
- **输出**: 24步 × 3维 轨迹 (dx, dy, delta_yaw)
- **模型**: TransformerActionHead 或 DiffusionActionHead
- **GT**: 归一化的相对运动序列

### 3.2 采用的 Loss

根据配置使用以下两种之一：

**TransformerActionHead (推荐)**:
```python
# 使用交叉熵或 MSE 计算动作预测损失
result = model.transformer_action_head.compute_loss(pred_actions, gt_actions)
trajectory_loss = result['loss']
```

**DiffusionActionHead**:
```python
# 标准 DDPM 噪声预测 MSE
noise = torch.randn_like(gt_trajectory)
timesteps = torch.randint(0, num_train_timesteps, (batch_size,))
noisy_trajectory = noise_scheduler.add_noise(gt_trajectory, noise, timesteps)
noise_pred = model.forward_diffusion(noisy_trajectory, timesteps, cond)
loss = F.mse_loss(noise_pred, noise)
```

**公式**:
$$L_{trajectory} = \mathbb{E}_{t, \epsilon} \left[ ||\epsilon - \epsilon_\theta(x_t, t, c)||^2 \right]$$

### 3.3 设计理由

| 选择 | 理由 |
|------|------|
| Diffusion | 生成多样轨迹，处理多模态输出 |
| MSE on noise | 标准 DDPM 训练目标，稳定收敛 |
| 无加权 | 轨迹各点同等重要，无类别不平衡 |

---

## 4. 进度预测 Loss

### 4.1 任务特点

- **输入**: Qwen3-VL 特征 (1024维)
- **输出**: 1维标量 [0, 1]，表示导航进度
- **模型**: ProgressHead (MLP)
- **GT分布**: 均匀分布，但边界点 (0 和 1) 是关键决策点

### 4.2 采用的 Loss: 简单 MSE

```python
# 简单 MSE，无加权
loss = F.mse_loss(progress, targets)
```

**公式**:
$$L_{progress} = \frac{1}{N} \sum_{i} (p_i - \hat{p}_i)^2$$

### 4.3 网络结构

```python
# 3 层 MLP (对齐 InternNav DistanceNetwork)
nn.Sequential(
    nn.Linear(input_dim, input_dim // 4),
    nn.ReLU(),
    nn.Linear(input_dim // 4, input_dim // 16),
    nn.ReLU(),
    nn.Linear(input_dim // 16, 1),
    nn.Sigmoid(),
)
```

### 4.4 设计理由

| 选择 | 理由 |
|------|------|
| 简单 MSE | 对齐 InternNav，progress 均匀分布无需加权 |
| 3 层 MLP | InternNav 验证过的简单有效结构 |
| ReLU | 简单激活函数，避免过拟合 |
| 无 LayerNorm/Dropout | 简单回归任务不需要复杂正则化 |

---

## 5. 停止预测 Loss

### 5.1 任务特点

- **输入**: Qwen3-VL 特征 (1024维)
- **输出**: 1维标量 (0 或 1)，表示是否停止
- **模型**: StopHead
- **GT**: 二进制标签

### 5.2 采用的 Loss: BCE

```python
# 二元交叉熵
loss = F.binary_cross_entropy_with_logits(stop_logit, stop_target)
```

---

## 6. 总体 Loss 组合

### 6.1 总 Loss 公式

$$L_{total} = \lambda_h \cdot L_{heatmap} + \lambda_t \cdot L_{trajectory} + \lambda_p \cdot L_{progress} + \lambda_s \cdot L_{stop}$$

### 6.2 当前权重配置

```yaml
loss:
  history_weight: 1.0       # λ_h (热力图)
  future_weight: 0.0        # λ_f (未来热力图，已关闭)
  trajectory_weight: 0.0    # λ_t (轨迹，已关闭)
  progress_weight: 0.0      # λ_p (进度，已关闭)
  stop_weight: 0.0          # λ_s (停止，已关闭)
```

**注意**：当前配置只训练热力图任务，其他任务头已关闭。

### 6.3 权重设计理由

| 任务 | 权重 | 理由 |
|------|------|------|
| 热力图 | 1.0 | 核心任务，输出维度最大，多重 loss 联合优化 |
| 轨迹 | 0.0 | 单独训练 |
| 进度 | 0.0 | 单独训练 |
| 停止 | 0.0 | 辅助任务，已被进度预测替代 |

---

## 7. 实际 Loss 输出监控

训练时监控以下 loss 分量：

```python
{
    'loss': total_loss,                    # 总 loss
    'diffusion_loss': diffusion_loss,      # 纯扩散 loss（80% focal）
    'base_loss': base_loss,                # 标准 MSE (20%)
    'focal_loss': focal_loss,              # 空间 focal MSE (80%)
    'x0_loss': x0_loss,                    # x0 重构损失 (×3.0)
    'dice_loss': dice_loss,                 # Dice Loss (×2.0，新增)
    'sparsity_loss': sparsity_loss,        # 稀疏性损失 (×0.1)
    'neg_zero_loss': neg_zero_loss,        # 负样本零目标损失 (×0.5)
    'peak_dist_loss': peak_dist_loss,      # 峰值距离损失 (×5.0)
    'visibility_loss': visibility_loss,    # 可见性 BCE loss (×0.5)
    'action_loss': action_loss,            # 轨迹/动作 loss
    'stop_loss': stop_loss,                # 停止 loss
}
```

---

## 8. Loss 设计演进总结

### v1 (旧版)
- focal_weight: 0.3
- x0_loss_weight: 1.0
- 无 Dice Loss
- sparsity_loss_weight: 0.5
- peak_distance_loss_weight: 2.0

### v2 (新版)
- focal_weight: **0.8** (aggressive)
- x0_loss_weight: **3.0** (主力)
- **新增 Dice Loss (×2.0)**
- sparsity_loss_weight: **0.1** (降低)
- peak_distance_loss_weight: **5.0** (加强)
- **新增 positive_sample_boost: 3.0**

### 核心改进
1. **双重 Focal 机制**：样本级 (正样本 3x) + 空间级 (峰值 15-21x)
2. **Dice Loss**：专治稀疏信号，比 MSE 更适合热力图
3. **x0 主力化**：x0_loss 成为主要优化目标
4. **更激进的峰值定位**：peak_distance_loss_weight 提升到 5.0

---

## 9. 配置参考

```yaml
model:
  heatmap_head:
    use_circular_padding: true   # 360° 全景图支持
    use_visibility_head: true    # 可见性预测头

    positive_sample_boost: 3.0   # 正样本权重提升
    negative_sample_weight: 0.3  # 负样本降权

    x0_loss_weight: 3.0          # x0 重构损失 (主力)
    dice_loss_weight: 2.0         # Dice Loss (新增)
    sparsity_loss_weight: 0.1     # 稀疏性损失
    visibility_loss_weight: 0.5   # 可见性损失
    peak_distance_loss_weight: 5.0 # 峰值距离损失

  action_head:
    type: transformer           # Transformer DDPM (推荐)

  progress_head:
    hidden_dim: 512

loss:
  history_weight: 1.0
  future_weight: 0.0            # 关闭未来热力图
  trajectory_weight: 0.0       # 单独训练
  progress_weight: 0.0          # 单独训练
  stop_weight: 0.0              # 关闭
```

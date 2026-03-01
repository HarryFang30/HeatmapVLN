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

### 2.2 采用的 Loss: 混合 Diffusion + 辅助损失

```python
# 1. 主损失：标准 DDPM MSE + Focal 峰值加权
per_pixel_mse = (noise_pred - noise) ** 2
base_loss = (sample_weight * per_pixel_mse).mean()

# Focal 损失：让模型更关注峰值区域
gt_weight = gt_heatmap.clamp(0, 1)
weight_map = 1.0 + focal_alpha * gt_weight
focal_loss = (sample_weight * weight_map * per_pixel_mse).mean()

focal_weight = 0.3
diffusion_loss = (1 - focal_weight) * base_loss + focal_weight * focal_loss
```

**公式**:
$$L_{diffusion} = \mathbb{E}_{t, \epsilon} \left[ ||\epsilon - \epsilon_\theta(x_t, t, c)||^2 \right]$$

### 2.3 辅助损失

```python
# 2. x0 重构损失：从噪声预测反推 x0 估计，直接监督输出质量
alpha_bar = self.noise_scheduler.alphas_cumprod[timesteps]
x0_hat = (noisy_heatmap - sqrt(1-alpha_bar) * noise_pred) / sqrt(alpha_bar)
x0_loss = MSE(x0_hat, gt_normalized)

# 3. L1 稀疏性正则化：鼓励输出大部分为 0
x0_denorm = (x0_hat + 1) / 2  # 还原到 [0, 1]
sparsity_loss = x0_denorm.mean()

# 4. 负样本显式零目标损失 (SNR 门控)
# 对负样本（GT=全零），约束 x0 应为全 -1
neg_zero_loss = MSE(neg_x0, -1.0)

# 5. 多峰感知峰值距离损失 (SNR 门控)
# 检测 GT 中的所有峰值，计算预测位置与 GT 的 L2 距离
peak_dist_loss = multi_peak_distance(snr_x0, snr_gt)
```

### 2.4 可见性预测损失

```python
# 可见性预测头：判断当前视角是否能看到历史点
gt_has_peak = (gt_heatmap.max() > 0.01).float()
vis_logit = visibility_head(cond)
visibility_loss = F.binary_cross_entropy_with_logits(vis_logit, gt_has_peak)
```

### 2.5 负样本处理

```python
# 检测负样本（GT 全零，即不可见目标）
is_negative = (gt_heatmap.max() < 0.01).float()

# 负样本扩散 loss 降权
sample_weight = 1.0 - (1.0 - negative_sample_weight) * is_negative
```

### 2.6 360° 全景图支持 (Circular Padding)

```python
# UNet Conv 层使用循环填充
# 水平方向: circular padding (左右连续)
# 垂直方向: replicate padding (上下不连续)

x = F.pad(x, (1, 1, 0, 0), mode='circular')  # 水平
x = F.pad(x, (0, 0, 1, 1), mode='replicate') # 垂直
```

配置启用：
```yaml
model:
  heatmap_head:
    use_circular_padding: true
```

### 2.7 为什么使用 Diffusion？

对于可能出现 **多峰分布** 的热力图（1个峰、3个峰、7个峰都可能），Diffusion 是最佳选择：

| 方案 | 单峰 | 多峰 | 原因 |
|------|------|------|------|
| CNN + KL | ✅ | ❌ | 多峰时学到"糊状"输出 |
| CNN + 固定K点 | ❌ | ❌ | 峰数量不匹配 |
| **Diffusion** | ✅ | ✅ | 去噪过程自然"雕刻"任意数量的峰 |

### 2.8 设计理由

| 选择 | 理由 |
|------|------|
| DDPM MSE + Focal | 基础扩散损失保证收敛，Focal 聚焦峰值区域 |
| x0 重构损失 | 直接监督输出质量，补充噪声预测 |
| 稀疏性正则化 | 鼓励大部分像素为 0，符合热力图分布 |
| 峰值距离损失 | 可微分地约束峰值位置准确 |
| SNR 门控 | 高时间步 x0 估计质量差，梯度有害，动态跳过 |
| 可见性头 | 显式建模"是否可见"，消除假阳性 |
| Circular Padding | 360° 全景图左右边界连续 |

### 2.9 当前权重配置

```yaml
model:
  heatmap_head:
    use_circular_padding: true   # 360° 全景图支持
    x0_loss_weight: 1.0           # x0 重构损失权重
    sparsity_loss_weight: 0.5     # 稀疏性损失权重
    visibility_loss_weight: 0.5  # 可见性损失权重
    negative_sample_weight: 0.3   # 负样本降权
    peak_distance_loss_weight: 2.0 # 峰值距离损失权重

loss:
  history_weight: 1.0
  future_weight: 1.0
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
  future_weight: 1.0        # λ_f (未来热力图)
  trajectory_weight: 1.0    # λ_t
  progress_weight: 0.5      # λ_p
  stop_weight: 0.5          # λ_s
```

### 6.3 权重设计理由

| 任务 | 权重 | 理由 |
|------|------|------|
| 热力图 | 1.0 | 核心任务，输出维度最大 |
| 轨迹 | 1.0 | 核心任务，直接影响导航 |
| 进度 | 0.5 | 辅助任务，相对简单 |
| 停止 | 0.5 | 辅助任务，二分类简单 |

---

## 7. 实际 Loss 输出监控

训练时监控以下 loss 分量：

```python
{
    'loss': total_loss,                    # 总 loss
    'heatmap_loss': heatmap_loss,          # 热力图噪声预测 loss
    'diffusion_loss': diffusion_loss,      # 纯扩散 loss（不含 visibility）
    'visibility_loss': visibility_loss,    # 可见性 BCE loss
    'base_loss': base_loss,                # 标准 MSE
    'focal_loss': focal_loss,              # 峰值加权 MSE
    'x0_loss': x0_loss,                    # x0 重构损失
    'sparsity_loss': sparsity_loss,        # 稀疏性损失
    'neg_zero_loss': neg_zero_loss,       # 负样本零目标损失
    'peak_dist_loss': peak_dist_loss,      # 峰值距离损失
    'action_loss': action_loss,            # 轨迹/动作 loss
    'stop_loss': stop_loss,                # 停止 loss
}
```

---

## 8. 配置参考

```yaml
model:
  heatmap_head:
    use_circular_padding: true   # 360° 全景图支持
    use_visibility_head: true    # 可见性预测头
    x0_loss_weight: 1.0
    sparsity_loss_weight: 0.5
    visibility_loss_weight: 0.5
    negative_sample_weight: 0.3
    peak_distance_loss_weight: 2.0

  action_head:
    use_diffusion: true          # 使用 DiffusionActionHead

  progress_head:
    hidden_dim: 256

  stop_head:
    hidden_dim: 256

loss:
  history_weight: 1.0
  future_weight: 1.0
  trajectory_weight: 1.0
  progress_weight: 0.5
  stop_weight: 0.5
```

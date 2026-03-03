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

### 2.3 Min-SNR 时间步加权 (ICCV 2023)

```python
# 计算每个时间步的信噪比 (SNR)
alpha_bar = noise_scheduler.alphas_cumprod[timesteps]  # (B,)
snr = alpha_bar / (1 - alpha_bar)                      # (B, 1, 1, 1)

# Min-SNR 加权：clamp(SNR, max=gamma) / SNR
min_snr_gamma = 5.0
snr_weight = torch.clamp(snr, max=min_snr_gamma) / snr  # (B, 1, 1, 1)
```

**核心思想**：标准 DDPM 对所有时间步等权重训练，但不同时间步的学习效率差异巨大：

| 时间步区间 | SNR | 噪声预测难度 | 权重行为 |
|-----------|-----|-------------|---------|
| 低噪声步 (t < 50) | SNR > 5 (高) | 简单 | **压低** (snr_weight < 1)，减少梯度浪费 |
| 中噪声步 | 0.5 ≤ SNR ≤ 5 | 适中 | **保持** (snr_weight ≈ 1) |
| 高噪声步 (t > 120) | SNR < 0.5 (低) | 困难但对推理链至关重要 | **保持全权重** (snr_weight = 1) |

**参考论文**: "Efficient Diffusion Training via Min-SNR Weighting Strategy" (ICCV 2023)

### 2.4 主损失：Epsilon MSE + Min-SNR + 样本权重

```python
# Epsilon Loss（噪声预测损失）
per_pixel_mse = (noise_pred - noise) ** 2  # (B, 1, H, W)

# 综合加权：Min-SNR 时间步权重 × 正负样本权重
diffusion_loss = (snr_weight * sample_weight_4d * per_pixel_mse).mean()

total_loss = diffusion_loss
```

**公式**:
$$L = \mathbb{E}_{t, \epsilon} \left[ w_{snr}(t) \cdot w_{sample} \cdot ||\epsilon - \epsilon_\theta(x_t, t, c)||^2 \right]$$

其中：
- $w_{snr}(t) = \min(\text{SNR}(t), \gamma) / \text{SNR}(t)$，$\gamma = 5.0$
- $w_{sample}$ = 正样本 boost × is_positive + 负样本 weight × is_negative

**设计理由**：
- 移除了所有辅助损失（x0、dice、sparsity、neg_zero、peak_distance、visibility）
- 仅保留 **单一 epsilon MSE**，通过 Min-SNR 加权实现高效训练
- 简化后的 loss 更稳定，避免多 loss 之间的权重调参和梯度冲突

### 2.5 为什么使用 Diffusion？

对于可能出现 **多峰分布** 的热力图（1个峰、3个峰、7个峰都可能），Diffusion 是最佳选择：

| 方案 | 单峰 | 多峰 | 原因 |
|------|------|------|------|
| CNN + KL | ✅ | ❌ | 多峰时学到"糊状"输出 |
| CNN + 固定K点 | ❌ | ❌ | 峰数量不匹配 |
| **Diffusion** | ✅ | ✅ | 去噪过程自然"雕刻"任意数量的峰 |

### 2.6 完整 Loss 组合

```
Total Loss = diffusion_loss
           = (Min-SNR权重 × 样本权重 × epsilon_MSE).mean()
```

无辅助损失，单一目标函数。

### 2.7 当前权重配置

```yaml
model:
  heatmap_head:
    use_circular_padding: true   # 360° 全景图支持
    positive_sample_boost: 3.0   # 正样本权重提升
    negative_sample_weight: 0.3  # 负样本降权

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

训练时监控以下指标：

```python
{
    'loss': total_loss,                    # 总 loss = diffusion_loss
    'diffusion_loss': diffusion_loss,      # Min-SNR 加权 epsilon MSE
    'eps_mse_high_snr': float,             # 低噪声区 (SNR>5) 的 epsilon MSE
    'eps_mse_mid_snr': float,              # 中噪声区 (0.5≤SNR≤5) 的 epsilon MSE
    'eps_mse_low_snr': float,              # 高噪声区 (SNR<0.5) 的 epsilon MSE
    'heatmap': pred_heatmap,               # 推理热力图（每 100 步生成一次）
    'noise_pred': noise_pred,              # 模型预测的噪声
    'noise_target': noise,                 # 真实噪声
    'noise_std': float,                    # 真实噪声的标准差
    'noise_pred_std': float,               # 预测噪声的标准差
}
```

**SNR 分段诊断**：通过对比三个 SNR 区间的 epsilon MSE，可以判断模型在不同噪声水平下的去噪能力：
- `eps_mse_high_snr` 高 → 模型在简单（低噪声）时间步表现差
- `eps_mse_low_snr` 高 → 模型在困难（高噪声）时间步表现差（正常，这些步本身困难）
- `noise_std` vs `noise_pred_std` → 预测噪声的尺度是否与真实噪声匹配

---

## 8. Loss 设计演进总结

### v1
- focal_weight: 0.3
- x0_loss_weight: 1.0
- 无 Dice Loss
- sparsity_loss_weight: 0.5
- peak_distance_loss_weight: 2.0

### v2
- focal_weight: 0.8 (aggressive)
- x0_loss_weight: 3.0 (主力)
- 新增 Dice Loss (×2.0)
- sparsity_loss_weight: 0.1
- peak_distance_loss_weight: 5.0
- 新增 positive_sample_boost: 3.0
- 双重 Focal 机制 + 6 个辅助 loss

### v3 (当前版本)
- **移除所有辅助 loss**（x0、dice、sparsity、neg_zero、peak_distance、visibility）
- **移除空间 Focal 权重**
- **新增 Min-SNR 时间步加权** (gamma=5.0)
- 仅保留 epsilon MSE + 样本权重 + Min-SNR 加权
- 新增 SNR 分段诊断监控

### 核心改进 (v2 → v3)
1. **极简化**：从 7 个 loss 分量精简为 **单一 loss**，消除多 loss 权重调参问题
2. **Min-SNR 加权**：让模型在不同时间步上获得更均匀的学习信号，替代手工设计的 focal/SNR 门控
3. **训练稳定性**：移除辅助 loss 的梯度冲突，收敛更稳定
4. **诊断升级**：通过 SNR 分段 epsilon MSE 监控模型在不同噪声水平的表现

---

## 9. 配置参考

```yaml
model:
  heatmap_head:
    use_circular_padding: true   # 360° 全景图支持
    positive_sample_boost: 3.0   # 正样本权重提升
    negative_sample_weight: 0.3  # 负样本降权

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

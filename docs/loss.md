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

- **输入**: Qwen3-VL 提取的视觉语言特征 (1024维)
- **输出**: 64×64 热力图，表示位置概率分布
- **GT特点**: 
  - 高斯分布形状（可能多峰，取决于导航场景）
  - ~93.5% 是黑色背景 (值接近0)
  - ~6.5% 是峰值区域 (中心最大值~1.0)
- **特殊处理**: 360° 全景图，左右边界连续

### 2.2 采用的 Loss: 标准 DDPM 无加权 MSE

```python
# 标准 DDPM Loss：无加权 MSE
# Diffusion 的去噪过程自然会学习数据分布
# 不需要人为约束（peak_loss/variance_loss 反而有害）

diffusion_loss = F.mse_loss(noise_pred, noise)
```

**公式**:
$$L_{diffusion} = \mathbb{E}_{t, \epsilon} \left[ ||\epsilon - \epsilon_\theta(x_t, t, c)||^2 \right]$$

### 2.3 为什么使用 Diffusion 而不是 CNN + KL？

对于可能出现 **多峰分布** 的热力图（1个峰、3个峰、7个峰都可能），Diffusion 是最佳选择：

| 方案 | 单峰 | 多峰 | 原因 |
|------|------|------|------|
| CNN + KL | ✅ | ❌ | 多峰时学到"糊状"输出 |
| CNN + 固定K点 | ❌ | ❌ | 峰数量不匹配 |
| **Diffusion** | ✅ | ✅ | 去噪过程自然"雕刻"任意数量的峰 |

### 2.4 360° 全景图支持 (Circular Padding)

由于输入是 Equirectangular 投影的 360° 全景图，左右边界在空间上是连续的：

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

### 2.5 设计理由

| 选择 | 理由 |
|------|------|
| 标准无加权 MSE | 噪声 ε 在空间上均匀分布，加权会破坏这个假设 |
| 移除 peak_loss | `max()` 只给 1 个像素梯度，导致随机噪点 |
| 移除 variance_loss | 模型学会输出高频噪声来满足约束 |
| Circular Padding | 360° 全景图左右边界连续 |

### 2.6 之前有害设计的问题

```python
# ❌ 已删除的有害设计

# 1. 加权 MSE - 破坏噪声均匀性假设
weight = 1.0 + 9.0 * gt_heatmap  # 峰值区域权重 x10
# 问题：模型在峰值区域预测过大噪声，去噪时崩溃

# 2. 峰值约束 - 只提供单点梯度
peak_loss = F.relu(0.3 - pred_heatmap.max())
# 问题：模型随便拉高一个像素满足约束 → 噪点

# 3. 方差约束 - 鼓励高频噪声
variance_loss = F.relu(0.05 - pred_heatmap.std())
# 问题：输出白噪声是最简单的高方差输出
```

---

## 3. 轨迹预测 Loss

### 3.1 任务特点

- **输入**: Qwen3-VL 特征 (1024维)
- **输出**: 24步 × 3维 轨迹 (dx, dy, delta_yaw)
- **模型**: Transformer Decoder + Diffusion
- **GT**: 归一化的相对运动序列

### 3.2 采用的 Loss: 标准 DDPM 噪声预测 MSE

```python
# 采样噪声
noise = torch.randn_like(gt_trajectory)

# 采样时间步
timesteps = torch.randint(0, num_train_timesteps, (batch_size,))

# 加噪
noisy_trajectory = noise_scheduler.add_noise(gt_trajectory, noise, timesteps)

# 预测噪声
noise_pred = model.forward_diffusion(noisy_trajectory, timesteps, cond)

# 标准 MSE Loss
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
- **GT分布**: 均匀分布，但边界点 (0 和 1) 是关键决策点

### 4.2 采用的 Loss: 简单 MSE (对齐 InternNav)

```python
# 简单 MSE，无加权
loss = F.mse_loss(progress, targets)
```

**公式**:
$$L_{progress} = \frac{1}{N} \sum_{i} (p_i - \hat{p}_i)^2$$

### 4.3 网络结构 (对齐 InternNav DistanceNetwork)

```python
# 简单 3 层 MLP
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

### 4.5 为什么不用边界加权

之前使用 `boundary_weight = 1 + 2 * |target - 0.5|` 导致 loss 不收敛。

**原因分析**：
- Progress 是均匀分布的 (0 → 1)
- 边界加权让 0 和 1 附近样本权重翻倍
- 导致训练在边界和中间之间"拉扯"，无法稳定收敛
- InternNav 验证了简单 MSE 足够有效

---

## 5. 总体 Loss 组合

### 5.1 总 Loss 公式

$$L_{total} = \lambda_h \cdot L_{heatmap} + \lambda_t \cdot L_{trajectory} + \lambda_p \cdot L_{progress}$$

### 5.2 当前权重配置

```yaml
loss:
  history_weight: 1.0      # λ_h
  trajectory_weight: 1.0   # λ_t
  progress_weight: 0.5     # λ_p
```

### 5.3 权重设计理由

| 任务 | 权重 | 理由 |
|------|------|------|
| 热力图 | 1.0 | 核心任务，输出维度最大 |
| 轨迹 | 1.0 | 核心任务，直接影响导航 |
| 进度 | 0.5 | 辅助任务，相对简单 |

---

## 6. 修复历史

### 6.1 2024-01 Loss 重构

**问题**: 热力图输出为噪点，模型"作弊"满足约束而不学习分布

**原因分析**:
1. `peak_loss = F.relu(0.3 - max())` - `max()` 只给 1 个像素梯度
2. `variance_loss = F.relu(0.05 - std())` - 高频噪声是最简单的高方差输出
3. 加权 MSE 破坏噪声均匀性假设

**修复**:
- ✅ 移除 `peak_loss` 和 `variance_loss`
- ✅ 使用标准无加权 MSE: `F.mse_loss(noise_pred, noise)`
- ✅ 添加 360° Circular Padding 支持

---

## 7. 配置参考

```yaml
model:
  heatmap_head:
    use_circular_padding: true   # 360° 全景图支持
    dropout: 0.2

loss:
  heatmap_loss_type: simplified  # 后处理 loss（可选）
  history_weight: 1.0
  trajectory_weight: 1.0
  progress_weight: 0.5
```

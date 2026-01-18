# HeatmapVLN

本目录实现了一个用于 **第一人称跨帧热力图（inter-frame heatmap）** 与 **动作预测** 的训练/评估/推理流水线。

以下是设计架构图：

<img src="assets/architecture.png" width="800">

*N帧照片构成的视频序列，与当前观测，动作指令一起输入LLM，LLM输出token通过重排生成二维向量，最终通过ConditionalUnet2D生成热力图。我们只要求模型在当前观测的Nk帧中能准确抓住空间关系以解决时间累计造成的数据量溢出的问题。最终我们希望生成的热力图能为导航提供重要的位置信息以供参考。*


当前仓库内真实可用的入口脚本为：

- `scripts/train.py`：训练（单阶段训练 history 热力图 + action + stop）
- `scripts/evaluate.py`：评估（支持 history/action，支持保存可视化）
- `scripts/inference.py`：推理（支持对视频或数据集 clip 运行并保存热力图/动作）


---

## 快速开始

### 1) 环境安装

建议在 `HeatmapVLN` 目录下安装依赖：

```bash
cd HeatmapVLN

# 建议 Python 3.11+（与本项目依赖/代码路径更匹配）
conda create -n models python=3.11 -y
conda activate models

pip install -U pip
pip install -r requirements.txt
```

如果你需要安装带 CUDA 的 PyTorch，请按你机器 CUDA 版本选择对应 wheel（`requirements.txt` 内也有注释提示）。

---


## 数据集加载逻辑（VLNSlidingWindowDataset）

### 核心思想：Clip-level 采样（推荐）

为解决滑动窗口采样导致的**样本高度相关性**问题（相邻帧生成的样本非常相似，容易过拟合），我们采用 **Clip-level 采样策略**：

**每个 epoch 从每个 clip 随机选择 N 个样本，而不是使用所有滑动窗口样本**

| 采样方式 | 样本数（假设1400 clips） | 样本相关性 | 每 epoch 变化 |
|---------|-------------------------|-----------|---------------|
| 滑动窗口(stride=1) | ~133,000 | 极高（>90%帧重叠） | 固定 |
| 滑动窗口(stride=5) | ~26,600 | 高（相邻帧相似） | 固定 |
| **Clip-level(N=2)** | ~2,800/epoch | **低** | **每 epoch 重新采样** |

**为什么 clip-level 采样有效？**
- 单 epoch 样本数虽少，但 50 epochs 累计可看到 ~140,000 个不同样本
- 每 epoch 从每个 clip 随机选择不同时刻，增加数据多样性
- 避免相邻帧高度相关导致的过拟合

### Clip-level 采样流程

```
Epoch 1:                          Epoch 2:
  Clip A → [STOP帧] + 随机选帧[12]    Clip A → [STOP帧] + 随机选帧[8]
  Clip B → [STOP帧] + 随机选帧[15]    Clip B → [STOP帧] + 随机选帧[23]
  ...                               ...（不同的随机组合）
```

**关键特性**：
- 每个 clip 的**最后一帧（STOP）始终被采样**，确保 STOP 样本不丢失
- 从非 STOP 帧中随机选择 `samples_per_clip - 1` 个样本
- 通过 `set_epoch()` 触发重新采样，每个 epoch 看到不同的样本组合
- 验证集使用固定滑动窗口采样（禁用 clip-level），确保指标可比

### 数据增强

训练集自动启用数据增强（仅影响图像，不影响热力图和动作标签）：

| 增强类型 | 参数 | 说明 |
|---------|------|------|
| **ColorJitter** | brightness=0.3, contrast=0.3, saturation=0.2, hue=0.1, p=0.5 | 颜色抖动，增加光照鲁棒性 |
| **GaussianNoise** | std=8.0, p=0.3 | 高斯噪声，增加噪声鲁棒性 |

### 关键参数

```python
VLNSlidingWindowDataset(
    root="/path/to/dataset",
    split="train",
    min_history=5,              # 最小历史帧数（T >= 5 才生成样本）
    num_history_sample=8,       # 从历史中采样的帧数 K
    image_size=(224, 224),      # 图像尺寸
    hm_size=(64, 64),           # 热力图尺寸
    sample_stride=5,            # 采样步长（滑动窗口模式使用）
    # Clip-level 采样配置（防止过拟合的关键）
    clip_level_sampling=True,   # 启用 clip-level 采样（推荐）
    samples_per_clip=2,         # 每个 clip 每 epoch 采样数（1 STOP + N-1 正常）
    enable_augmentation=True,   # 启用数据增强（仅训练集）
)
```

### 配置文件

```yaml
data:
  sliding_window:
    min_history: 5
    num_history_sample: 8
    sample_stride: 5            # 滑动窗口模式的步长
    # Clip-level 采样（解决样本高度相关性问题，防止过拟合）
    clip_level_sampling: true   # 启用（训练集自动使用，验证集自动禁用）
    samples_per_clip: 2         # 每 clip 每 epoch 采样 2 个样本（1 STOP + 1 正常）
```

### 采样数量估算

| 配置 | 单 epoch 样本数 | 50 epochs 累计 | 说明 |
|------|----------------|----------------|------|
| `samples_per_clip=2` | clips × 2 | clips × 100 | 推荐，平衡效率与覆盖 |
| `samples_per_clip=3` | clips × 3 | clips × 150 | 更多覆盖，略慢 |
| `samples_per_clip=1` | clips × 1 | clips × 50 | 最快，可能欠拟合 |

### 训练时调用

```python
# 每个 epoch 开始时调用，触发重新采样
train_loader.dataset.set_epoch(epoch)
```

### 返回格式

```python
{
    "history_frames": [K, 3, H, W],    # K 帧历史（均匀采样）
    "current_frame": [3, H, W],        # 当前观测
    "heatmap": [Hm, Wm],               # 历史位置热力图
    "action": [2],                     # 连续动作 (dx, dy)
    "action_valid": float,             # 是否有效（最后一帧=0）
    "discrete_action": int,            # 离散动作 (0=STOP, 1=FORWARD, 2=LEFT, 3=RIGHT)
    "is_stop": float,                  # 是否 STOP (0 or 1)
    "text": str,                       # 导航指令
}
```

### 热力图生成

热力图通过 **3D → 2D 投影 + 高斯模糊** 生成：

1. 读取历史帧和当前帧的 **相机位姿**（4×4 矩阵）
2. 将历史相机中心转换到当前相机坐标系
3. 投影到 Equirectangular 图像坐标
4. 绘制 **自适应高斯点**（距离越远，sigma 越小）
5. （可选）使用 **深度图** 进行遮挡检测

---

## 热力图生成模块（DiffusionHeatmapHead）

热力图生成模块使用 **条件扩散模型** 从 LLM 特征和视觉观测中生成空间热力图。

### 架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        DiffusionHeatmapHead 架构                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐                                         │
│  │ LLM Tokens  │    │  观测帧     │                                         │
│  │[B,seq,2048] │    │[B,3,H,W]    │                                         │
│  └──────┬──────┘    └──────┬──────┘                                         │
│         │                  │                                                │
│         ▼                  ▼                                                │
│  ┌─────────────┐    ┌─────────────┐                                         │
│  │ Attention   │    │ CNN Encoder │  ← ImageConditionEncoder (可选)         │
│  │  Pooling    │    │ (GroupNorm) │    use_image_encoder=True/False         │
│  │ (可学习Q)   │    │[32,64,128,  │                                         │
│  └──────┬──────┘    │   256]通道  │                                         │
│         │           └──────┬──────┘                                         │
│         ▼                  │                                                │
│  ┌─────────────┐           │                                                │
│  │LLMCondition │           │    ┌─────────────────────────────────┐         │
│  │ Projector   │           │    │ LLM-only 模式 (推荐)            │         │
│  │ +Dropout    │           │    │ use_image_encoder=False         │         │
│  └──────┬──────┘           │    │ 只使用 LLM 特征，无 CNN 编码    │         │
│         │                  │    └─────────────────────────────────┘         │
│         └───────┬──────────┘                                                │
│                 ▼                                                           │
│          ┌─────────────┐                                                    │
│          │ Concat + MLP│  ← MultiModalConditionEncoder                      │
│          │  (融合层)   │    LLM+Image 或 LLM-only                           │
│          │  +Dropout   │    输出 [B, cond_dim]                              │
│          └──────┬──────┘                                                    │
│                 │                                                           │
│     ┌───────────┴───────────┐                                               │
│     │                       │                                               │
│     ▼                       ▼                                               │
│ ┌────────┐           ┌─────────────┐                                        │
│ │条件向量│──────────▶│ConditionalU │                                        │
│ │[B,512] │  global   │   Net2D     │  ← FiLM 条件调制                       │
│ └────────┘   cond    │ (噪声预测)  │    Sinusoidal Timestep Embedding       │
│                      └──────┬──────┘                                        │
│                             │                                               │
│                 ┌───────────┴───────────┐                                   │
│                 │                       │                                   │
│                 ▼                       ▼                                   │
│          ┌─────────────┐         ┌─────────────┐                            │
│          │  Noisy HM   │  ←───── │ DDPM 10步   │                            │
│          │[B,1,Hm,Wm]  │  迭代   │  采样器     │                            │
│          └─────────────┘  去噪   └──────┬──────┘                            │
│                                         │                                   │
│                                         ▼                                   │
│                                  ┌─────────────┐                            │
│                                  │  Heatmap    │  对数空间归一化            │
│                                  │ [B,Hm,Wm]   │  → [0, 1]                  │
│                                  └─────────────┘                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 模块组件

#### 1. 条件编码器（MultiModalConditionEncoder）

负责融合文本和视觉信息，支持两种模式：

**LLM + Image 模式**（`use_image_encoder=True`）：

| 组件 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `LLMConditionProjector` | [B, seq, 2048] | [B, cond_dim] | AttentionPool → Linear → Dropout → LayerNorm → GELU → Linear |
| `ImageConditionEncoder` | [B, 3, H, W] | [B, cond_dim] | 轻量级 CNN (Stem + 3 Stages + GAP + Projection) |
| `Fusion MLP` | [B, cond_dim×2] | [B, cond_dim] | Concat → Linear → Dropout → LayerNorm → GELU → Linear |

**LLM-only 模式**（`use_image_encoder=False`，推荐）：

| 组件 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `LLMConditionProjector` | [B, seq, 2048] | [B, cond_dim] | AttentionPool → Linear → Dropout → LayerNorm → GELU → Linear |
| `Fusion MLP` | [B, cond_dim] | [B, cond_dim] | Linear → Dropout → LayerNorm → GELU → Linear |

> **消融实验结论**：LLM-only 模式效果更好（Val Loss 下降 12.5%），因为 Qwen3-VL 已经处理了当前帧，CNN 重复编码反而增加过拟合风险。

#### 2. Attention Pooling（序列聚合）

使用可学习的 query 向量通过 attention 机制聚合 LLM 序列特征：

```python
class AttentionPooling(nn.Module):
    """
    比 mean pooling 更好地保留空间和语义信息
    
    Args:
        dim: 特征维度 (2048)
        num_heads: 注意力头数 (默认 4)
    """
    # 可学习的 query 向量
    self.query = nn.Parameter(torch.randn(1, 1, dim) * 0.02)
    
    # 多头注意力聚合
    # query: (B, 1, dim) 与 key/value: (B, seq_len, dim) 计算 attention
    # 输出: (B, dim) 聚合后的特征
```

**池化方法选项**：
| 方法 | 说明 | 推荐场景 |
|------|------|---------|
| `attention` | 可学习 query + 多头注意力（默认） | 最佳效果，推荐使用 |
| `mean` | 平均池化 | 简单场景 |
| `last` | 取最后一个 token | 自回归模型 |
| `first` | 取第一个 token | CLS token 场景 |
| `max` | 最大池化 | 快速实验 |

#### 3. ImageConditionEncoder（观测图像编码器）

轻量级 CNN 编码器，使用 **GroupNorm** 替代 BatchNorm 以提高小 batch 稳定性：

```python
# 架构（使用 GroupNorm）
Stem:     Conv 7×7 stride 2 → GroupNorm → ReLU → MaxPool
Stage 1:  ConvBlock(32→64, stride=2) + ResidualBlock(64)
Stage 2:  ConvBlock(64→128, stride=2) + ResidualBlock(128)
Stage 3:  ConvBlock(128→256, stride=2) + ResidualBlock(256)
Pool:     Global Average Pooling → [B, 256]
Project:  Linear(256, cond_dim) → Dropout → LayerNorm → GELU → Linear → Dropout
```

**参数量**：约 2.3M（可通过 `use_image_encoder=False` 禁用）

#### 4. 噪声预测网络（ConditionalUnet2D）

基于 2D U-Net 的条件去噪网络，使用 FiLM 调制：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `in_channels` | 1 | 输入通道（热力图单通道） |
| `out_channels` | 1 | 输出通道（预测噪声） |
| `block_out_channels` | (64, 128, 256) | 各层通道数 |
| `layers_per_block` | 2 | 每层残差块数量 |
| `attention_levels` | (2,) | 添加注意力的层级（最深层） |
| `dropout` | 0.1 | Dropout 正则化 |

**架构详情**：
```
Encoder: [ConditionalResidualBlock2D × 2 + Downsample2D] × 3 levels
Middle:  ConditionalResidualBlock2D → Attention2D → ConditionalResidualBlock2D
Decoder: [Upsample2D + ConditionalResidualBlock2D × 2 + Skip Connection] × 3 levels
Output:  GroupNorm → SiLU → Conv2d
```

**FiLM 条件调制**（Feature-wise Linear Modulation）：
```python
# 时间步嵌入 + 全局条件
cond = time_embed(timestep) + global_cond  # (B, cond_dim)

# 预测 scale 和 shift
scale, shift = MLP(cond).chunk(2)  # 各 (B, channels)

# 调制特征
h = h × (1 + scale) + shift
```

#### 5. 扩散调度器（DDPMScheduler）

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `num_train_timesteps` | 100 | 训练扩散步数 |
| `num_inference_steps` | 10 | 推理采样步数（可调，更少更快） |
| `beta_schedule` | `squaredcos_cap_v2` | 余弦噪声调度（比 linear 更平滑） |
| `prediction_type` | `epsilon` | 预测噪声（非直接预测样本） |
| `clip_sample` | True | 采样时裁剪到合理范围 |

### 数据流

```
                    ┌─────────────────────────────────────────┐
                    │         LLM-only 模式（推荐）           │
                    └─────────────────────────────────────────┘
文本流: LLM Token [B,seq,2048] → Attention Pool → Projection → [B, cond_dim]
                                                                    ↓
条件流: [B, cond_dim] → Fusion MLP → [B, cond_dim] → ConditionalUnet2D
                                                                    ↓
生成流: 随机噪声 [B,1,Hm,Wm] → 迭代去噪 (10步) → 对数反归一化 → Heatmap [B,Hm,Wm]

                    ┌─────────────────────────────────────────┐
                    │         LLM + Image 模式                │
                    └─────────────────────────────────────────┘
文本流: LLM Token [B,seq,2048] → Attention Pool → Projection → [B, cond_dim]
                                                                    ↓
视觉流: 观测帧 [B,3,H,W] → CNN Encoder → [B, cond_dim] ────────→ Concat
                                                                    ↓
条件流: [B, cond_dim×2] → Fusion MLP → [B, cond_dim] → ConditionalUnet2D
                                                                    ↓
生成流: 随机噪声 [B,1,Hm,Wm] → 迭代去噪 (10步) → 对数反归一化 → Heatmap [B,Hm,Wm]
```

### 对数空间归一化

热力图使用对数空间归一化，更好地保留峰值信息：

```python
# 归一化（训练时）
def _normalize_heatmap(heatmap):
    # 1. Max-to-1 归一化
    heatmap_norm = heatmap / heatmap.max()
    
    # 2. 对数变换：让信号分布更均匀
    log_scale = 6.0
    log_heatmap = torch.log(heatmap_norm * log_scale + 1)
    
    # 3. 归一化到 [-1, 1]
    normalized = (log_heatmap / log(log_scale + 1)) * 2 - 1
    return normalized

# 反归一化（推理时）
def _denormalize_heatmap(heatmap):
    # 逆对数变换 + clamp 到 [0, 1]
    recovered = (exp((heatmap + 1) / 2 * log(7)) - 1) / 6
    return recovered.clamp(0, 1)
```

**为什么用对数归一化**：
- 原始热力图 93.5% 是 0（背景），直接归一化后信号集中在 -1 附近
- 对数变换让信号分布更均匀，扩散模型更容易学习

### 训练与推理

**训练模式**（加权 MSE + 峰值保持损失）：

```python
# 1. 前向扩散：给 GT 热力图加噪
gt_normalized = normalize_heatmap(gt_heatmap)  # 对数空间归一化
noisy_heatmap = scheduler.add_noise(gt_normalized, noise, timesteps)

# 2. 预测噪声
noise_pred = unet(noisy_heatmap, timesteps, global_cond)

# 3. 加权 MSE Loss：峰值区域权重 x10
weight = 1.0 + 9.0 * gt_heatmap.clamp(0, 1)  # [1.0, 10.0]
diffusion_loss = (weight * (noise_pred - noise).pow(2)).mean()

# 4. 峰值保持损失：确保输出不是全黑
pred_heatmap = diffusion_inference(cond)
peak_loss = F.relu(0.3 - pred_heatmap.max())      # 最大值必须 >= 0.3
variance_loss = F.relu(0.05 - pred_heatmap.std())  # 必须有空间变化

# 5. 总损失
loss = diffusion_loss + 1.0 * (peak_loss + variance_loss)
```

**推理模式**：

```python
# 从纯噪声开始
noisy_heatmap = torch.randn(B, 1, Hm, Wm)

# 设置推理步数
scheduler.set_timesteps(num_inference_steps=10)

# 迭代去噪
for t in scheduler.timesteps:
    noise_pred = unet(noisy_heatmap, t, global_cond)
    noisy_heatmap = scheduler.step(noise_pred, t, noisy_heatmap).prev_sample

# 对数反归一化到 [0, 1]
heatmap = denormalize_heatmap(noisy_heatmap).squeeze(1)  # [B, Hm, Wm]
```

### 配置参数

在 `configs/train_config.yaml` 中：

```yaml
model:
  heatmap_head:
    enable_history: true          # 启用历史热力图头
    enable_future: false          # 禁用未来热力图头（可选）
    cond_dim: 512                 # 条件向量维度
    num_inference_steps: 10       # 推理扩散步数
    use_image_encoder: false      # 推荐 false（LLM-only 模式）
    llm_pool_method: attention    # 推荐 attention pooling
    dropout: 0.1                  # Dropout 正则化
```

**DiffusionHeatmapConfig 完整参数**：

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `llm_dim` | 2048 | Qwen3-VL hidden dimension |
| `cond_dim` | 512 | 条件向量维度 |
| `heatmap_size` | (64, 64) | 输出热力图尺寸 |
| `use_image_encoder` | True | 是否使用 CNN 编码观测（推荐 False） |
| `llm_pool_method` | 'attention' | LLM 序列池化方法 |
| `llm_pool_num_heads` | 4 | Attention pooling 头数 |
| `block_out_channels` | (64, 128, 256) | UNet 各层通道数 |
| `num_train_timesteps` | 100 | 训练扩散步数 |
| `num_inference_steps` | 10 | 推理扩散步数 |
| `dropout` | 0.1 | Dropout 正则化率 |

### 源代码位置

| 文件 | 说明 |
|------|------|
| `src/models/heatmap/__init__.py` | 模块导出 |
| `src/models/heatmap/diffusion_heatmap_head.py` | 主模块：`DiffusionHeatmapHead`, `create_diffusion_heatmap_head` |
| `src/models/heatmap/diffusion/config.py` | 配置：`DiffusionHeatmapConfig` |
| `src/models/heatmap/diffusion/unet2d.py` | 噪声预测：`ConditionalUnet2D`, `ConditionalResidualBlock2D`, `Attention2D` |
| `src/models/heatmap/diffusion/image_encoder.py` | 条件编码：`MultiModalConditionEncoder`, `LLMConditionProjector`, `ImageConditionEncoder`, `AttentionPooling` |

---

## 动作预测模块

动作预测模块提供多种动作生成方式，包括基于扩散模型的连续动作生成和离散动作分类。

### 模块概览

| 组件 | 类型 | 输出 | 说明 |
|------|------|------|------|
| `DiffusionActionHead` | 连续动作 | (dx, dy) | 基于 1D U-Net 扩散模型，生成 2D 位移 |
| `TransformerActionHead` | 连续轨迹 | (x, y, θ) × T | 基于 Transformer Decoder 扩散模型，生成多步轨迹 |
| `StopPredictionHead` | 二分类 | STOP/继续 | 独立的停止动作预测器 |
| `ProgressPredictionHead` | 回归 | 0-1 | 任务完成进度预测，替代 STOP 分类 |
| `DiscreteActionHead` | 多分类 | 4 类 | STOP/FORWARD/LEFT/RIGHT 离散动作 |

---

### DiffusionActionHead（2D 连续动作）

使用 **条件扩散模型** 从 LLM 特征中生成导航动作（2D 连续位移）。

#### 架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                       DiffusionActionHead 架构                              │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐                                                            │
│  │ LLM Tokens  │                                                            │
│  │ [B,seq,2048]│                                                            │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────┐                                                            │
│  │  Mean Pool  │  ← ConditionProjector                                     │
│  │  [B, 2048]  │    Pool + Linear + LayerNorm + GELU + Dropout             │
│  └──────┬──────┘    + Linear + LayerNorm + Dropout                          │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────┐                                                            │
│  │ Projection  │                                                            │
│  │  [B, 256]   │  ← 条件向量（encoding_size=256，简化架构）                │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         └─────────────────────┐                                             │
│                               │                                             │
│                               ▼                                             │
│                        ┌─────────────┐                                      │
│                        │Conditional  │  ← ConditionalResidualBlock1D        │
│         ┌─────────────▶│   Unet1D    │    FiLM 条件调制                     │
│         │              │(噪声预测)  │    down_dims=[128,256]               │
│         │              └──────┬──────┘                                      │
│         │                     │                                             │
│         │         ┌───────────┴───────────┐                                 │
│         │         │                       │                                 │
│         │         ▼                       ▼                                 │
│  ┌──────┴─────┐ ┌─────────────┐   ┌─────────────┐                          │
│  │条件向量    │ │ Noisy Action│   │ DDPM 10步   │                          │
│  │[B,256]    │ │ [B,1,2]     │◀──│  采样器     │                          │
│  └───────────┘ └─────────────┘   └──────┬──────┘                          │
│                   迭代去噪              │                                   │
│                                         ▼                                   │
│                                  ┌─────────────┐                            │
│                                  │  Actions    │  加权 MSE Loss             │
│                                  │  [B,1,2]    │  + 方差约束                │
│                                  └──────┬──────┘                            │
│                                         │                                   │
│                                         ▼                                   │
│                                  ┌─────────────┐                            │
│                                  │Unnormalize  │                            │
│                                  │& Cumsum     │                            │
│                                  └──────┬──────┘                            │
│                                         │                                   │
│                                         ▼                                   │
│                                  ┌─────────────┐                            │
│                                  │ Final Action│                            │
│                                  │   (dx, dy)  │                            │
│                                  └─────────────┘                            │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 条件投影器（ConditionProjector）

将 LLM 特征投影为扩散模型的条件向量，增加 Dropout 正则化：

```python
class ConditionProjector(nn.Module):
    def __init__(self, input_dim=2048, output_dim=256, dropout=0.2):
        self.projector = nn.Sequential(
            nn.Linear(input_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout),      # 第一层 Dropout
            nn.Linear(output_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(dropout),      # 第二层 Dropout
        )
```

#### 噪声预测网络（ConditionalUnet1D）

基于 1D U-Net 的条件去噪网络，使用 FiLM 调制：

```
架构：
  Encoder: [ConditionalResidualBlock1D × 2] × N levels
  Middle:  ConditionalResidualBlock1D × 2
  Decoder: [ConditionalResidualBlock1D × 2 + Skip Connection] × N levels
  Output:  Conv1dBlock → Conv1d

FiLM 调制:
  cond = SinusoidalPosEmb(timestep) + global_cond
  if cond_predict_scale:
      out = scale * out + bias  # scale, bias from MLP(cond)
  else:
      out = out + bias          # bias only
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `input_dim` | 2 | 输入维度（dx, dy） |
| `global_cond_dim` | 256 | 全局条件维度 |
| `down_dims` | [128, 256] | U-Net 通道数（简化架构） |
| `diffusion_step_embed_dim` | 256 | 时间步嵌入维度 |
| `kernel_size` | 3 | 1D 卷积核大小 |
| `n_groups` | 8 | GroupNorm 分组数 |
| `dropout` | 0.1 | Dropout 正则化 |

#### 动作归一化（ActionStats）

动作被归一化到 [-1, 1] 范围：

```python
@dataclass
class ActionStats:
    min: List[float] = [-0.5, -0.2]  # 允许后退和左转
    max: List[float] = [0.5, 1.0]    # 允许前进和右转

# 归一化公式
normalized = (action - min_val) / (max_val - min_val) * 2.0 - 1.0

# 反归一化公式
action = (normalized + 1.0) / 2.0 * (max_val - min_val) + min_val
```

#### 训练与推理

**训练模式（加权 MSE + 方差约束）**：

```python
# 1. 归一化 GT 动作到 [-1, 1]
normalized_gt = normalize_actions(gt_actions, action_stats)

# 2. 前向扩散：给 GT 动作加噪
noisy_actions = scheduler.add_noise(normalized_gt, noise, timesteps)

# 3. 预测噪声
noise_pred = unet(noisy_actions, timesteps, global_cond)

# 4. 加权 MSE Loss：非零动作权重更高
# 问题：95.5% 的转向动作是 0，模型学会"输出 0"就能获得极低 Loss
# 解决：增加非零动作的权重
action_magnitude = normalized_gt.abs()
weight = 1.0 + 9.0 * action_magnitude.clamp(0, 1)  # [1, 10]
diffusion_loss = (weight * (noise_pred - noise).pow(2)).mean()

# 5. 方差约束：防止输出全零
variance_loss = F.relu(0.1 - pred_actions.std())
loss = diffusion_loss + 0.3 * variance_loss
```

**推理模式**：

```python
# 从纯噪声开始
noisy_actions = torch.randn(B, pred_horizon, action_dim)

# 迭代去噪
for t in scheduler.timesteps:
    noise_pred = unet(noisy_actions, t, global_cond)
    noisy_actions = scheduler.step(noise_pred, t, noisy_actions).prev_sample

# 反归一化 + 累积求和得到位置
actions = unnormalize_actions(noisy_actions, action_stats)
positions = torch.cumsum(actions, dim=1)  # delta → position
```

---

### TransformerActionHead（多步轨迹生成）

参考 **InternNav** 实现，使用 Transformer Decoder + Diffusion Policy 生成多步导航轨迹。

#### 架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    TransformerActionHead 架构                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐                                                            │
│  │ LLM Tokens  │                                                            │
│  │[B,seq,1024] │  ← vlm_token_dim (可配置)                                  │
│  └──────┬──────┘                                                            │
│         │                                                                   │
│         ▼                                                                   │
│  ┌─────────────┐    ┌─────────────┐                                        │
│  │ Mean Pool   │    │ Time Embed  │                                        │
│  │  [B,1,1024] │    │ [B,1,384]   │  ← SinusoidalPosEmb                    │
│  └──────┬──────┘    └──────┬──────┘                                        │
│         │                  │                                                │
│         ▼                  │                                                │
│  ┌─────────────┐           │                                                │
│  │ cond_obs_emb│           │                                                │
│  │  [B,1,384]  │ ──────────┼────────────────────────────┐                  │
│  └──────┬──────┘           │                            │                  │
│         │                  │                            │                  │
│         └────────┬─────────┘                            │                  │
│                  ▼                                      │                  │
│           ┌─────────────┐                               │                  │
│           │   Concat    │                               │                  │
│           │ [B,2,384]   │ ← T_cond = 1 + n_obs_steps   │                  │
│           └──────┬──────┘                               │                  │
│                  │                                      │                  │
│                  ▼                                      │                  │
│           ┌─────────────┐                               │                  │
│           │ Transformer │ ← n_cond_layers=4            │                  │
│           │  Encoder    │   条件编码器                  │                  │
│           └──────┬──────┘                               │                  │
│                  │                                      │                  │
│                  │   memory                             │                  │
│                  ▼                                      ▼                  │
│           ┌─────────────┐                        ┌─────────────┐           │
│           │ Transformer │◀───────────────────────│Noisy Action │           │
│           │  Decoder    │  cross-attention       │[B,24,3]     │           │
│           │ n_layer=16  │  + causal mask         │(x,y,θ)×T    │           │
│           └──────┬──────┘                        └─────────────┘           │
│                  │                                      ▲                  │
│                  ▼                                      │                  │
│           ┌─────────────┐                               │                  │
│           │  Output MLP │                               │                  │
│           │ [B,24,3]    │  ← noise_pred                │                  │
│           └──────┬──────┘                               │                  │
│                  │                                      │                  │
│                  └──────────────────────────────────────┘                  │
│                           DDPM 迭代去噪                                     │
│                                                                             │
│                                  ▼                                          │
│                           ┌─────────────┐                                   │
│                           │ Trajectory  │                                   │
│                           │ [B,24,3]    │  ← (x, y, theta) × 24 steps      │
│                           └─────────────┘                                   │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

#### 关键特性

| 特性 | 说明 |
|------|------|
| **完整权重初始化** | 参考 InternNav `_init_weights`，正态分布初始化 |
| **条件编码器** | TransformerEncoder 预处理条件（n_cond_layers=4） |
| **因果掩码** | `tgt_mask` + `memory_mask` 确保因果性 |
| **位置嵌入** | 正态分布初始化的可学习嵌入 |
| **多步预测** | predict_size=24 步轨迹 |

#### 配置参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `vlm_token_dim` | 1024 | VLM 输出维度 |
| `n_emb` | 384 | 内部嵌入维度 |
| `predict_size` | 24 | 预测步数 |
| `n_layer` | 16 | Decoder 层数 |
| `n_head` | 8 | 注意力头数 |
| `n_cond_layers` | 4 | Encoder 层数 |
| `action_dim` | 3 | 动作维度 (x, y, θ) |
| `num_train_timesteps` | 20 | 扩散训练步数 |

---

### StopPredictionHead（停止动作预测）

独立的二分类器，判断是否应该执行 STOP 动作。使用 **混合 Loss** 处理极度类别不平衡（STOP 样本仅 3%）。

#### 架构

```python
class StopPredictionHead(nn.Module):
    def __init__(self, input_dim=2048, hidden_dim=512, dropout=0.1):
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1),  # Binary output
        )
```

#### 混合 Loss（BCE + Focal）

```python
# 问题：纯 Focal Loss 在极度不平衡时过度压低梯度
# 解决：混合 Loss = 0.3 * BCE + 0.7 * Focal

# 1. 类别权重：STOP 类权重 10x
pos_weight = torch.tensor([10.0])
bce_loss = F.binary_cross_entropy_with_logits(logits, targets, pos_weight=pos_weight)

# 2. Focal Loss（限制 gamma）
gamma = min(self.focal_gamma, 2.0)  # 限制最大 gamma=2
focal_weight = (1 - p_t) ** gamma
focal_loss = alpha_weight * focal_weight * bce_loss

# 3. 混合
loss = 0.3 * bce_loss + 0.7 * focal_loss
```

---

### ProgressPredictionHead（任务进度预测）

替代二分类的 STOP 预测，使用连续值 (0-1) 表示任务完成进度。参考 **InternNav** 的 `pg_pred_mlp`。

#### 优势

| 对比项 | StopPredictionHead | ProgressPredictionHead |
|--------|-------------------|------------------------|
| 输出类型 | 二分类 (0/1) | 连续值 (0-1) |
| 损失函数 | Focal + BCE | MSE + 边界增强 |
| 监督信号 | 只有终点有信号 | 每步都有进度信号 |
| 中间状态 | 无法表达 | 可表达（如 0.8 = "快到了"） |

#### 架构

```python
class ProgressPredictionHead(nn.Module):
    def __init__(self, input_dim=1024, hidden_dim=512, dropout=0.1):
        self.progress_mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, hidden_dim // 4),
            nn.GELU(),
            nn.Linear(hidden_dim // 4, 1),
            nn.Sigmoid(),  # 输出 0-1
        )
```

#### 边界增强 Loss

```python
# 对接近 0 或 1 的样本给予更高权重（开始和停止是关键决策点）
boundary_weight = 1.0 + 2.0 * torch.abs(targets - 0.5)  # [1, 2]
loss = (mse_loss * boundary_weight).mean()
```

#### 使用

```python
progress_head = ProgressPredictionHead(input_dim=2048)

# 训练
loss = progress_head(llm_features, gt_progress=gt_progress, return_loss=True)['loss']

# 推理
progress = progress_head.get_progress(llm_features)  # (B,) in [0, 1]
should_stop = progress_head.predict_stop(llm_features, threshold=0.9)  # (B,) binary
```

---

### DiscreteActionHead（离散动作分类）

可选的全离散动作分类器，预测 {STOP, FORWARD, LEFT, RIGHT}。

```python
class DiscreteActionHead(nn.Module):
    ACTION_NAMES = ['STOP', 'FORWARD', 'LEFT', 'RIGHT']
    
    def __init__(self, input_dim=2048, hidden_dim=512, num_actions=4):
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, num_actions),
        )
```

---

### 配置参数

在 `configs/train_config.yaml` 中：

```yaml
model:
  action_head:
    enable: true                    # 启用动作预测
    action_dim: 2                   # 2D 动作 (dx, dy)
    pred_horizon: 1                 # 预测步数
    num_diffusion_iters: 10         # 扩散步数
    encoding_size: 256              # 条件维度
    down_dims: [128, 256]           # U-Net 通道数
    action_stats_min: [-0.5, -0.2]  # 归一化最小值
    action_stats_max: [0.5, 1.0]    # 归一化最大值

  stop_head:
    enable: true                    # 启用 Stop 预测
    hidden_dim: 512
    focal_gamma: 2.0
    focal_alpha: 0.75

loss:
  action_weight: 1.0                # 动作损失权重
  stop_weight: 0.5                  # Stop 损失权重
```

### 源代码位置

| 文件 | 说明 |
|------|------|
| `src/models/action/__init__.py` | 模块导出 |
| `src/models/action/diffusion_action_head.py` | `DiffusionActionHead`, `ConditionProjector` |
| `src/models/action/transformer_action_head.py` | `TransformerActionHead` (InternNav 风格) |
| `src/models/action/stop_head.py` | `StopPredictionHead`, `DiscreteActionHead` |
| `src/models/action/progress_head.py` | `ProgressPredictionHead` |
| `src/models/action/action_config.py` | `DiffusionActionConfig` |
| `src/models/action/utils.py` | `ActionStats`, `normalize_actions`, `unnormalize_actions` |
| `src/models/action/diffusion/conditional_unet1d.py` | `ConditionalUnet1D`, `ConditionalResidualBlock1D` |
| `src/models/action/diffusion/conv1d_components.py` | `Conv1dBlock`, `Downsample1d`, `Upsample1d` |
| `src/models/action/diffusion/positional_embedding.py` | `SinusoidalPosEmb` |

---

## Qwen3-VL 集成模块

本项目使用 **Qwen3-VL** 作为视觉语言骨干网络，替代传统的 VGGT + DINOv3 组合。该模块封装了 Qwen3-VL 的加载、推理和特征提取功能。

### 架构概览

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        Qwen3-VL Integration 架构                            │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                     │
│  │ History     │    │  Current    │    │ Instruction │                     │
│  │ Video Frames│    │  Frame      │    │   Text      │                     │
│  │[B,K,C,H,W]  │    │[B,C,H,W]    │    │   string    │                     │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                     │
│         │                  │                  │                            │
│         └──────────────────┴──────────────────┘                            │
│                            │                                               │
│                            ▼                                               │
│                    ┌───────────────┐                                       │
│                    │  Qwen3-VL     │  ← Flash Attention 2                  │
│                    │  Processor    │    (apply_chat_template)              │
│                    └───────┬───────┘                                       │
│                            │                                               │
│                            ▼                                               │
│                    ┌───────────────┐                                       │
│                    │  Qwen3-VL     │  ← 参数冻结 (不训练)                   │
│                    │  Model        │    output_hidden_states=True          │
│                    └───────┬───────┘                                       │
│                            │                                               │
│            ┌───────────────┼───────────────┐                               │
│            ▼               ▼               ▼                               │
│     ┌─────────────┐ ┌─────────────┐ ┌─────────────┐                       │
│     │Hidden States│ │ Vision      │ │ Generated   │                       │
│     │[B,seq,2048] │ │ Hidden      │ │ Text        │                       │
│     └──────┬──────┘ │[B,V,2048]   │ │ (optional)  │                       │
│            │        └──────┬──────┘ └─────────────┘                       │
│            │               │                                               │
│            └───────┬───────┘                                               │
│                    │                                                       │
│                    ▼                                                       │
│            ┌─────────────────────────────────────┐                         │
│            │         Downstream Heads            │                         │
│            ├─────────────┬───────────┬───────────┤                         │
│            │ Heatmap     │  Action   │   Stop    │                         │
│            │   Head      │   Head    │   Head    │                         │
│            └─────────────┴───────────┴───────────┘                         │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 核心组件

#### 1. Qwen3VLConfig（配置类）

```python
@dataclass
class Qwen3VLConfig:
    model_path: str = "./models/qwen_3_vl"      # 模型路径
    device: str = "cuda"                         # 设备
    torch_dtype: str = "bfloat16"               # 数据类型
    attn_implementation: str = "flash_attention_2"  # 推荐使用 FA2
    max_video_frames: int = 16                   # 最大视频帧数
    hidden_layer_for_features: int = -1          # 提取哪一层 hidden states
    
    # Sequence Packing 设置
    enable_packing: bool = False                 # 是否启用 Sequence Packing
    max_seq_length: int = 4096                   # 最大打包序列长度
    spatial_merge_size: int = 2                  # 视觉空间合并大小
```

#### 2. Qwen3VLIntegration（主集成类）

| 方法 | 输入 | 输出 | 说明 |
|------|------|------|------|
| `forward()` | history_frames, current_frame, instruction | hidden_states, vision_hidden_states | 标准批量推理 |
| `forward_packed()` | packed_batch | hidden_states, vision_hidden_states, seq_lens | Sequence Packing 推理 |
| `get_data_collator()` | - | FlattenedDataCollatorForVLN | 获取 Packing 专用 Collator |
| `enable_sequence_packing()` | - | bool | 启用 varlen attention |

**关键特性**：
- **参数冻结**：Qwen3-VL 所有参数冻结，不参与训练
- **批量处理**：支持 left padding 的真正批量推理
- **梯度流动**：虽然参数冻结，但保留计算图以便梯度回传到下游模块

#### 3. Sequence Packing（高效批量训练）

基于 Qwen3-VL 官方 fine-tuning 框架实现，显著提高显存利用率。

**传统 Padding vs Sequence Packing**：

```
传统 Padding（浪费显存）：
┌──────────────────────────────────────────────┐
│ [样本A, PAD, PAD, PAD, PAD, PAD, PAD, PAD]   │  ← 50% 是 PAD
│ [样本B, PAD, PAD, PAD, PAD, PAD, PAD, PAD]   │  ← 50% 是 PAD
│ [样本C, PAD, PAD, PAD, PAD, PAD, PAD, PAD]   │  ← 50% 是 PAD
└──────────────────────────────────────────────┘

Sequence Packing（最大化利用）：
┌──────────────────────────────────────────────┐
│ [样本A, 样本B, 样本C, 样本D, 样本E, ...]     │  ← 无 PAD
└──────────────────────────────────────────────┘
  ↑      ↑      ↑
  cumsum_seq_lens = [0, len_A, len_A+len_B, ...]
```

**核心技术**：

| 技术 | 说明 |
|------|------|
| **Bin Packing** | 将多个变长样本打包到固定长度序列 |
| **Flattened Attention Mask** | 使用 cumulative sequence lengths 代替 2D mask |
| **flash_attn_varlen_func** | 高效处理变长序列的 FlashAttention |
| **3D RoPE Position IDs** | 正确处理视觉和文本的位置编码 |

#### 4. FlattenedDataCollatorForVLN

专门为 VLN 任务设计的 Data Collator：

```python
@dataclass
class FlattenedDataCollatorForVLN:
    tokenizer: Any
    
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        """
        输入：每个 instance 包含 input_ids, position_ids, pixel_values, ...
        输出：packed batch，包含：
            - input_ids: (1, total_seq_len)
            - attention_mask: cumsum_seq_lens, shape (num_samples + 1,)
            - position_ids: (3, 1, total_seq_len) for M-RoPE
            - seq_lens: List[int]，用于拆分 hidden states
        """
```

### 使用示例

#### 基本推理

```python
from src.models.qwen3_vl import Qwen3VLIntegration, Qwen3VLConfig

# 初始化
config = Qwen3VLConfig(model_path="./models/qwen_3_vl")
qwen_vl = Qwen3VLIntegration(config)

# 准备输入
history_frames = torch.randn(2, 8, 3, 224, 224)  # [B, K, C, H, W]
current_frame = torch.randn(2, 3, 224, 224)      # [B, C, H, W]
instruction = "Walk forward and turn left at the door."

# 推理
outputs = qwen_vl(history_frames, current_frame, instruction)
hidden_states = outputs["hidden_states"]          # [B, seq_len, 2048]
vision_hidden = outputs["vision_hidden_states"]   # [B, V, 2048]
```

#### Sequence Packing 模式

```python
from src.models.qwen3_vl import Qwen3VLIntegration, Qwen3VLConfig

# 启用 packing
config = Qwen3VLConfig(
    model_path="./models/qwen_3_vl",
    enable_packing=True,
    max_seq_length=4096,
)
qwen_vl = Qwen3VLIntegration(config)

# 获取 collator
collator = qwen_vl.get_data_collator()

# 准备样本（每个样本独立 tokenize）
samples = [preprocess_sample(s) for s in batch]
packed_batch = collator(samples)

# Packed 推理
outputs = qwen_vl.forward_packed(packed_batch)
sample_hidden = outputs["hidden_states"]       # [num_samples, hidden_dim]
vision_hidden = outputs["vision_hidden_states"]  # [num_samples, max_V, hidden_dim]
```

### 辅助函数

| 函数 | 说明 |
|------|------|
| `split_packed_hidden_states()` | 将 packed hidden states 拆分为各样本表示（支持 last/mean/first pooling） |
| `split_packed_vision_hidden_states()` | 提取各样本的视觉 token hidden states |
| `get_rope_index_3()` | 计算 Qwen3-VL 的 3D RoPE position IDs |
| `replace_attention_with_varlen()` | 替换 attention forward 以支持 varlen FlashAttention |

### 配置参数

在 `configs/train_config.yaml` 中：

```yaml
model:
  llm:
    model_path: ./models/qwen_3_vl        # Qwen3-VL 模型路径
    device: cuda
    torch_dtype: bfloat16
    attn_implementation: flash_attention_2  # 推荐使用 FA2
    max_video_frames: 16                    # 最大视频帧数（-1 表示不限制）
    hidden_layer_for_features: -1           # -1 = 最后一层
    
    # Sequence Packing（可选，高效训练）
    enable_packing: false                   # 是否启用
    max_seq_length: 4096                    # 最大打包序列长度
```

### 源代码位置

| 文件 | 说明 |
|------|------|
| `src/models/qwen3_vl/__init__.py` | 模块导出，PACKING_AVAILABLE 检查 |
| `src/models/qwen3_vl/integration.py` | 主模块：`Qwen3VLIntegration`, `Qwen3VLConfig` |
| `src/models/qwen3_vl/sequence_packing.py` | Packing 支持：`FlattenedDataCollatorForVLN`, `PackedSequenceProcessor` |

### 依赖要求

```bash
# 必需
pip install transformers>=4.45.0
pip install torch>=2.0

# 推荐（性能优化）
pip install flash-attn --no-build-isolation  # FlashAttention 2
```

> **注意**：如果未安装 `flash-attn`，Sequence Packing 的 varlen attention 将不可用，但基本功能仍可正常使用。

---

## 数据流架构（Sequence Packing Pipeline）

本项目的 Sequence Packing 实现符合 **Qwen3-VL 官方 fine-tuning 框架**，确保高效的多 batch 训练。

### 整体数据流

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                        完整数据流 Pipeline                                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  ┌─────────────────────┐                                                    │
│  │  VLNTrajectoryDataset│  ← 基础数据集                                      │
│  │  返回: history_frames,│    读取 RGB、位姿、动作等                          │
│  │  current_frame, text,│                                                   │
│  │  heatmap, trajectory │                                                   │
│  └──────────┬──────────┘                                                    │
│             │                                                               │
│             ▼                                                               │
│  ┌─────────────────────┐                                                    │
│  │ TokenizedVLNDataset │  ← 包装数据集（关键！）                             │
│  │  __getitem__() 中：  │    在 Dataset 层完成 tokenization                  │
│  │  1. 转换为 PIL 图像  │    可利用 DataLoader 的 num_workers 并行           │
│  │  2. 构建 messages    │                                                   │
│  │  3. processor.apply_ │                                                   │
│  │     chat_template()  │  ← Tokenization + 图像/视频处理                   │
│  │  4. 计算 position_ids│                                                   │
│  │                      │                                                   │
│  │  输出:               │                                                   │
│  │  - input_ids (1,seq) │                                                   │
│  │  - position_ids      │                                                   │
│  │  - pixel_values      │                                                   │
│  │  - pixel_values_videos│                                                  │
│  │  - VLN 数据 (heatmap,│                                                   │
│  │    trajectory, etc.) │                                                   │
│  └──────────┬──────────┘                                                    │
│             │                                                               │
│             │  num_workers > 0 时并行处理                                    │
│             ▼                                                               │
│  ┌─────────────────────┐                                                    │
│  │FlattenedCollatorFor │  ← 只做拼接，不做 tokenization                     │
│  │         VLN         │    符合官方 FlattenedDataCollatorForSupervisedDataset│
│  │                     │                                                    │
│  │  处理:              │                                                    │
│  │  1. torch.cat()     │  ← 拼接 input_ids, position_ids                    │
│  │     拼接序列        │                                                    │
│  │  2. 计算 cumsum_    │  ← attention_mask = [0, len1, len1+len2, ...]      │
│  │     seq_lens        │                                                    │
│  │  3. 拼接视觉数据    │  ← pixel_values, video_grid_thw 等                 │
│  │  4. Stack VLN 数据  │  ← current_frame, heatmap, trajectory              │
│  │                     │                                                    │
│  │  输出:              │                                                    │
│  │  - input_ids (1,    │                                                    │
│  │    total_seq_len)   │                                                    │
│  │  - attention_mask   │  ← cumsum_seq_lens (B+1,)                          │
│  │  - position_ids     │                                                    │
│  │  - seq_lens: List   │                                                    │
│  │  - num_samples: int │                                                    │
│  │  - VLN batch data   │                                                    │
│  └──────────┬──────────┘                                                    │
│             │                                                               │
│             ▼                                                               │
│  ┌─────────────────────┐                                                    │
│  │ model.forward_packed│  ← Qwen3-VL + 下游 Heads                           │
│  │      (batch)        │                                                    │
│  │                     │                                                    │
│  │  1. Qwen3-VL 处理   │  ← flash_attn_varlen_func                          │
│  │     packed 序列     │    使用 cumsum_seq_lens 作为 attention mask        │
│  │                     │                                                    │
│  │  2. split_packed_   │  ← 拆分 hidden states                              │
│  │     hidden_states() │    返回 (num_samples, hidden_dim)                  │
│  │                     │                                                    │
│  │  3. Heatmap Head    │  ← 使用 current_frame + hidden_states              │
│  │  4. Action Head     │  ← 使用 trajectory + hidden_states                 │
│  │  5. Progress Head   │  ← 使用 progress + hidden_states                   │
│  └─────────────────────┘                                                    │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 为什么在 Dataset 中 Tokenize？

官方 Qwen3-VL fine-tuning 框架的核心设计：

| 方式 | Tokenization 位置 | num_workers 效果 | 性能 |
|------|------------------|-----------------|------|
| ❌ 旧方式 | Collator 中 | 无法并行（主进程阻塞） | 慢 |
| ✅ **官方方式** | Dataset.__getitem__() 中 | **完全并行** | 快 3-10x |

**旧方式的问题**：
```python
# ❌ 旧方式：Collator 中 tokenize
class OldCollator:
    def __call__(self, samples):
        for sample in samples:
            # 这里在主进程串行执行，无法利用 num_workers
            result = processor.apply_chat_template(...)  # CPU 密集！
```

**官方方式（本项目采用）**：
```python
# ✅ 官方方式：Dataset 中 tokenize
class TokenizedVLNDataset(Dataset):
    def __getitem__(self, idx):
        sample = self.base_dataset[idx]
        # 这里在 worker 进程并行执行
        result = self.processor.apply_chat_template(...)  # 并行！
        return {**result, **sample}
```

### 数据格式验证

所有关键 tensor 的格式和维度：

```python
# TokenizedVLNDataset.__getitem__() 输出（单样本）
{
    "input_ids": (1, seq_len),           # int64
    "position_ids": (3, 1, seq_len),     # int64, 3D RoPE
    "attention_mask": [seq_len],         # List[int]
    "pixel_values": (N_patches, 1536),   # float32
    "image_grid_thw": (1, 3),            # int64
    "pixel_values_videos": (V_patches, 1536),
    "video_grid_thw": (1, 3),
    
    # VLN 数据
    "current_frame": (3, H, W),          # float32
    "heatmap": (Hm, Wm),                 # float32
    "trajectory": (24, 3),               # float32
    "trajectory_valid": float,
    "progress": float,
}

# FlattenedCollatorForVLN() 输出（packed batch）
{
    "input_ids": (1, total_seq_len),     # 拼接后
    "attention_mask": (num_samples + 1,), # cumsum_seq_lens, int32
    "position_ids": (3, 1, total_seq_len),
    "seq_lens": [seq1, seq2, ...],       # List[int]
    "num_samples": int,
    
    "pixel_values": (total_patches, 1536),
    "image_grid_thw": (num_samples, 3),
    "pixel_values_videos": (total_vid_patches, 1536),
    "video_grid_thw": (num_samples, 3),
    
    # VLN 数据（batched）
    "current_frame": (B, 3, H, W),
    "heatmap": (B, Hm, Wm),
    "trajectory": (B, 24, 3),
    "trajectory_valid": (B,),
    "progress": (B,),
}
```

### 性能对比

| num_workers | 首批时间 | 平均时间/batch | 相对加速 |
|-------------|---------|---------------|---------|
| 0 | 0.42s | 0.37s | 1x |
| 2 | 8.7s (初始化) | 0.20s | **1.9x** |
| 4 | 16.8s (初始化) | 0.04-0.10s | **3.5-10x** |

> 首批时间长是 `multiprocessing_context='spawn'` 模式的进程初始化开销（每个 worker 需要加载 processor），之后的 batch 会非常快。

### 源代码位置

| 文件 | 说明 |
|------|------|
| `src/data/tokenized_dataset.py` | `TokenizedVLNDataset`, `FlattenedCollatorForVLN` |
| `src/data/vln_sliding_window_dataset.py` | 基础数据集 `VLNTrajectoryDataset` |
| `scripts/train.py` | DataLoader 创建逻辑（自动选择 packing/非 packing 模式） |

### 配置示例

```yaml
# configs/train_config.yaml
model:
  llm:
    enable_packing: true          # 启用 Sequence Packing
    max_seq_length: 8192          # 最大打包序列长度
    spatial_merge_size: 2         # 视觉空间合并大小

data:
  num_workers: 4                  # 推荐 4-8
  prefetch_factor: 2              # 预取因子
```

### 梯度流动验证

所有可训练模块都能正确接收梯度：

```
✅ history_heatmap_head: grad_norm = 8.27
✅ transformer_action_head: grad_norm = 10.96
✅ progress_head: grad_norm = 2.89
✅ llm_projector: grad_norm = 10.65
✅ qwen3_vl (frozen): no gradients (正确)
```

---

## 数据集准备

### 数据集结构

训练/评估使用 `VLNSlidingWindowDataset`，配置文件通过 `data.root` 指定数据集根目录：

- 默认：`configs/train_config.yaml` 中的 `data.root: dataset_with_actions`
- Split: 训练用 `train`，验证用 `data.val_split`（如 `val_unseen`）

### 目录结构（示例）

```text
<data.root>/
  train/
    <scene_id>/
      clip_000000/
        meta.json                     # 必需：包含 num_frames, instruction
        poses.json                    # 必需：T 个 4×4 位姿矩阵
        rgb/                          # 必需：RGB 图像序列
          000000.png
          000001.png
          ...
        depth/                        # 可选：深度图（用于遮挡检测）
          000000.npy
          000001.npy
          ...
        actions.npy                   # 可选：连续动作 [T, 2] (dx, dy)
        discrete_actions.npy          # 可选：离散动作 [T] (0-3)
        intrinsics.json               # 可选：相机内参
      clip_000001/
        ...
  val_unseen/
    <scene_id>/
      clip_000000/
        ...
```

### 必需/可选文件说明
```
| 文件 | 类型 | 说明 |
|------|------|------|
| `meta.json` | 必需 | 至少包含 `num_frames`（帧数）；可包含 `instruction`（导航指令） |
| `poses.json` | 必需 | 长度为 T 的 4×4 位姿矩阵列表 |
| `rgb/000000.png` | 必需 | 按 6 位零填充命名的 RGB 帧 |
| `depth/000000.npy` | 可选 | 用于遮挡检测；缺失时跳过遮挡判断 |
| `actions.npy` | 可选 | 连续动作 [T, 2] (dx, dy)；缺失时 `action_valid=0` |
| `discrete_actions.npy` | 可选 | 离散动作 [T] (STOP/FORWARD/LEFT/RIGHT)；缺失时 stop 标签默认非 stop |
| `intrinsics.json` | 可选 | 相机内参；缺失时使用默认全景图尺寸 (512, 256) |
```
**建议**：训练 action/stop head 时务必提供对应的动作文件。

---

## 推理（Inference）

> ⚠️ **运行前请确保激活 conda 环境**：
> ```bash
> conda activate models
> ```

推理脚本：`scripts/inference.py`  
支持两种输入：

- **视频文件**：`--video /path/to/video.mp4`
- **数据集 clip 目录**：`--clip <data.root>/<split>/<scene>/clip_xxxxxx`

如果你不传 `--use-history/--use-future/--use-actions`，脚本会默认全部输出。

### 对数据集 clip 推理（推荐）

```bash
cd HeatmapVLN

python scripts/inference.py \
  --clip dataset_with_actions/val_unseen/<scene_id>/clip_000000 \
  --config configs/train_config.yaml \
  --output-dir ./outputs_inference
```

### 对视频推理

```bash
cd HeatmapVLN

python scripts/inference.py \
  --video /path/to/video.mp4 \
  --instruction "从起点出发，沿走廊前进并找到目标" \
  --config configs/train_config.yaml \
  --output-dir ./outputs_inference
```

### 推理输出

默认会在 `--output-dir` 下生成：

- `*_history_heatmaps.png`（若启用 history）
- `*_future_heatmaps.png`（若启用 future）
- `*_actions.npy`（若启用 action）

---

## 训练（Training）

> ⚠️ **运行前请确保激活 conda 环境**：
> ```bash
> conda activate models
> ```

训练脚本：`scripts/train.py`  
默认读取：`--config configs/train_config.yaml`

### 开始训练

```bash
cd HeatmapVLN

python scripts/train.py --config configs/train_config.yaml
```

### 常用训练参数

- **从检查点恢复**

```bash
python scripts/train.py \
  --config configs/train_config.yaml \
  --resume /path/to/ckpt.pth
```

- **自动从最新检查点恢复**

```bash
python scripts/train.py \
  --config configs/train_config.yaml \
  --auto-resume
```

- **调试：只构建模型与数据，不实际训练**

```bash
python scripts/train.py \
  --config configs/train_config.yaml \
  --dry-run

# 快速测试训练流程（每 epoch 只跑 5 个 batch）
python scripts/train.py \
  --config configs/train_config.yaml \
  --max-batches 5 \
  --epochs 2
```

**训练代码（后台运行）**
```bash
cd /root/HeatmapVLN && source /root/miniconda3/etc/profile.d/conda.sh && conda activate models && nohup python -u scripts/train.py --config configs/train_config.yaml > train.log 2>&1 &

# 实时查看日志
tail -f train.log
```

> 💡 **提示**：`python -u` 禁用输出缓冲，确保日志实时写入文件。

### 训练配置说明

配置文件 `configs/train_config.yaml` 包含单阶段训练配置:

| 名称 | Epochs | 分辨率 | 训练目标 | Loss 类型 |
|------|--------|--------|----------|-----------|
| `history_only_64` | 50 | 64×64 | History Head + Action Head + Stop Head | Simplified |

**关键配置项**:

```yaml
data:
  root: dataset_with_actions  # 数据集路径
  sliding_window:
    num_history_sample: 8     # 历史帧采样数
    sample_stride: 5          # 采样步长
    # Clip-level 采样（防止过拟合）
    clip_level_sampling: true # 启用 clip-level 采样
    samples_per_clip: 2       # 每 clip 每 epoch 采样 2 个样本

model:
  llm:
    model_path: ./models/qwen_3_vl  # Qwen3-VL 模型路径
  heatmap_head:
    enable_history: true      # 启用历史热力图头
    enable_future: false      # 禁用未来热力图头
  action_head:
    enable: true              # 启用动作预测
  stop_head:
    enable: true              # 启用 Stop 预测

optim:
  batch_size: 32              # 单卡 batch size
  grad_accum_steps: 4         # 梯度累积（有效 batch = 128）
  # 分组学习率（降低以防止过拟合）
  heatmap_lr: 1.0e-4          # 热力图头学习率
  action_lr: 1.0e-4           # 动作头学习率
  llm_projector_lr: 3.0e-5    # LLM 投影层学习率
  weight_decay: 1.0e-2        # 增加正则化

log:
  out_dir: vln_training_outputs
  use_tensorboard: true
```

### 防止过拟合策略

本项目采用多重策略防止模型过拟合：

| 策略 | 实现位置 | 说明 |
|------|---------|------|
| **Clip-level 采样** | Dataset | 每 epoch 从每个 clip 随机选 N 个样本，彻底解决滑动窗口样本高度相关性问题 |
| **每 epoch 重采样** | Dataset | 通过 `set_epoch()` 每 epoch 重建索引，保证看到不同样本组合 |
| **数据增强** | Dataset | ColorJitter(p=0.5) + GaussianNoise(p=0.3)，增加数据多样性 |
| **降低学习率** | Config | heatmap/action: 1e-4, projector: 3e-5 |
| **增加 weight_decay** | Config | 1e-2（增强 L2 正则化） |
| **Dropout 正则化** | 条件编码器 + UNet | ImageConditionEncoder、LLMConditionProjector、Fusion MLP、UNet 均使用 Dropout(0.1) |
| **GroupNorm 替代 BatchNorm** | ImageEncoder | 小 batch 下统计量更稳定，避免 BatchNorm 导致的训练不稳定 |
| **峰值/方差约束** | Diffusion Heads | 每 3 步检查输出，防止坍缩到全黑/全零 |

### CNN 消融实验

热力图头包含一个 `ImageConditionEncoder` (CNN) 用于编码当前观测。由于 Qwen3-VL 已经处理了当前帧，CNN 可能是冗余的。

**消融开关**：通过配置文件控制是否使用 CNN：

```yaml
model:
  heatmap_head:
    use_image_encoder: false  # 推荐设为 false（LLM-only），true 为 LLM+CNN
```

**实验结果**（2 epochs, 30 batches/epoch）：

| 配置 | Val Loss | 热力图 Loss | 动作 Loss | 结论 |
|------|----------|-------------|-----------|------|
| `use_image_encoder: true` | 2.5045 | 0.6032 | 1.7213 | 基线 |
| **`use_image_encoder: false`** | **2.1924** | **0.3477** | **1.6760** | **推荐** ✅ |

**关键发现**：
- **LLM-only 模式效果更好**：Val Loss 下降 12.5%，热力图 Loss 下降 42.4%
- **CNN 是冗余的**：Qwen3-VL 已经处理了当前帧，CNN 重复编码反而增加过拟合风险
- **建议保持 `use_image_encoder: false`**：移除 CNN 后模型更简洁，泛化能力更好，参数量减少约 2.3M

**运行消融实验**：

```bash
# LLM-only（推荐，当前默认）
python scripts/train.py --config configs/train_config.yaml --max-batches 100

# LLM + CNN（对比基线）
# 修改 configs/train_config.yaml 中 use_image_encoder: true
python scripts/train.py --config configs/train_config.yaml --max-batches 100
```

### Loss 设计

训练使用多任务联合 Loss：

$\mathcal{L} = \lambda_h \cdot \mathcal{L}_{heatmap} + \lambda_a \cdot \mathcal{L}_{action} + \lambda_s \cdot \mathcal{L}_{stop}$

默认权重：$\lambda_h=1.0$, $\lambda_a=1.0$, $\lambda_s=0.5$

| 任务 | Loss 类型 | 设计要点 |
|------|-----------|----------|
| 热力图 | Diffusion + 加权MSE | 峰值区域权重x10 + 峰值保持损失 + 方差约束 |
| 动作 | Diffusion + 加权MSE | 非零动作权重x10 + 方差约束 |
| 停止 | 混合Loss (BCE + Focal) | 10x类别权重 + gamma限制 |

---

#### Loss 收敛问题修复详解

**问题发现**：训练初期（<100步）所有 Loss 都异常快速下降到极低值：
- Heatmap Loss: 1.2 → 0.02
- Action Loss: 1.4 → 0.02  
- Stop Loss: 0.045（起点就很低）

**根因分析**：

| Loss | 数据特点 | 模型行为 |
|------|---------|---------|
| Heatmap | GT热力图93.5%是0（背景） | 输出全黑即可获得极低Loss |
| Action | 95.5%转向=0，40.3%前进=0 | 输出全零即可获得极低Loss |
| Stop | STOP样本仅3%（极度不平衡） | Focal Loss过度压低梯度 |

---

#### 1. Heatmap Loss 修复

**文件**: `src/models/heatmap/diffusion_heatmap_head.py`

**问题**：GT热力图极度稀疏（93.5%背景），普通MSE让模型学会"输出全黑"

**修复方案（三层防线）**：

```python
# 1. 加权 MSE Loss：峰值区域权重x10
weight = 1.0 + 9.0 * gt_heatmap.clamp(0, 1)  # [1, 10]
weighted_loss = (weight * squared_error).mean()

# 2. 峰值保持损失：确保输出有峰值
peak_loss = F.relu(0.3 - pred_heatmap.max())

# 3. 方差约束：确保输出有空间变化
variance_loss = F.relu(0.01 - pred_heatmap.std())

# 总损失
loss = diffusion_loss + 0.5 * (peak_loss + variance_loss)
```

**归一化改进**：使用对数变换让信号分布更均匀

```python
# 原归一化：[0,1] -> [-1,1]，导致93.5%的值都是-1
# 改进：对数空间归一化
log_heatmap = torch.log(heatmap * 6 + 1)  # [0, ~1.95]
normalized = (log_heatmap / max_log) * 2 - 1  # [-1, 1]
```

---

#### 2. Action Loss 修复

**文件**: `src/models/action/diffusion_action_head.py`

**问题**：动作数据分布极不均匀
- 维度0（转向）：95.5% 是 0
- 维度1（前进）：40.3% 是 0

**修复方案**：

```python
# 1. 加权 MSE Loss：非零动作权重更高
action_magnitude = normalized_gt.abs()
weight = 1.0 + 9.0 * action_magnitude.clamp(0, 1)  # [1, 10]
weighted_loss = (weight * squared_error).mean()

# 2. 方差约束：确保预测动作有变化
variance_loss = F.relu(0.1 - pred_actions.std())

# 总损失
loss = diffusion_loss + 0.3 * variance_loss
```

---

#### 3. Stop Loss 修复

**文件**: `src/models/action/stop_head.py`

**问题**：
- STOP 样本仅 3%（极度不平衡）
- 纯 Focal Loss 在此场景下过度压低梯度
- 初始 stop_loss = 0.045（正常BCE应为~0.69）

**修复方案**：

```python
# 1. 类别权重：STOP 类权重10x
pos_weight = torch.tensor([10.0], device=device)
bce_loss = F.binary_cross_entropy_with_logits(
    logits, targets, pos_weight=pos_weight
)

# 2. 限制 gamma 避免过度压低
gamma = min(self.focal_gamma, 2.0)

# 3. 混合 Loss = 0.3 * BCE + 0.7 * Focal
# BCE 保持基础梯度，Focal 聚焦困难样本
mixed_loss = 0.3 * bce_loss + 0.7 * focal_loss
```

---

#### 4. 总 Loss 权重调整

**文件**: `configs/train_config.yaml`

```yaml
loss:
  history_weight: 1.0   # Heatmap（核心任务）
  future_weight: 1.0    # Heatmap
  action_weight: 1.0    # Action（提升：0.5 → 1.0）
  stop_weight: 0.5      # Stop（内部已有10x类别权重）
```

---

#### 预期效果

| 指标 | 修复前 | 修复后预期 |
|------|--------|-----------|
| Heatmap Loss | 0.02 | 0.1-0.5 |
| Action Loss | 0.02 | 0.1-0.5 |
| Stop Loss | 0.045 | 0.3-0.5 |
| 热力图可视化 | 全黑 | 有明显峰值 |
| Stop 预测 | 全猜0.5 | 能区分STOP |

---

#### 诊断指标（TensorBoard）

修复后新增以下诊断指标：

```
diag/pred_heatmap_mean     # 预测热力图均值
diag/pred_heatmap_max      # 预测热力图最大值（<0.1说明坍缩）
diag/pred_heatmap_std      # 预测热力图标准差
diag/pred_heatmap_nonzero_ratio  # 非零像素比例
diag/heatmap_noise_std     # 热力图噪声标准差
diag/heatmap_noise_pred_std # 热力图噪声预测标准差
diag/action_noise_std      # 动作噪声标准差
diag/action_noise_pred_std # 动作噪声预测标准差
```

**坍缩检测**：如果 `pred_heatmap_max < 0.1`，日志会输出警告。

---

**分阶段策略**：Warmup 阶段使用 Diffusion Loss 稳定训练；后续阶段可切换到 NeRF 波纹 Loss 保留热力图高频细节。

### 输出文件结构

训练输出保存在 `log.out_dir` 指定路径:

```text
vln_training_outputs/
  ├── train.log                     # 训练日志
  ├── training_curves.png           # 训练曲线图（实时更新）
  ├── training_history.json         # 训练历史数据
  ├── best_model.pth                # 最佳模型
  ├── latest.pth                    # 最新检查点（用于续训）
  ├── history_only_64/              # 检查点目录
  │   ├── epoch_001.pth
  │   ├── epoch_002.pth
  │   └── ...
  └── visualizations/               # 热力图可视化
      └── epoch_001_step_00100.png
```

### 监控与调试

#### TensorBoard

```bash
tensorboard --logdir=/root/tf-logs --port=6006

# 查看指标:
# - train/loss, train/heatmap_loss, train/action_loss, train/stop_loss
# - val/loss, val/heatmap_loss, val/action_loss
# - train/lr, train/action_valid_ratio
# - train/heatmap_viz（热力图可视化）
```

#### 飞书通知

配置文件中启用飞书通知后，自动发送训练报告:

```yaml
log:
  notify:
    enabled: true
    platform: feishu
    webhook_url: "YOUR_WEBHOOK_URL"
```

---

## 常见问题 (FAQ)

### Q1: 显存不足 (CUDA Out of Memory)

**方案 1**: 减小 batch size

```yaml
optim:
  batch_size: 2          # 4 → 2
  grad_accum_steps: 8    # 保持有效 batch = 16
```

**方案 2**: 减少每 clip 采样数（推荐，使用 clip-level 采样时）

```yaml
data:
  sliding_window:
    samples_per_clip: 1    # 2 → 1，样本数减半
```

### Q2: 训练速度慢

**优化**: 减少每 clip 采样数

```yaml
data:
  sliding_window:
    samples_per_clip: 1    # 每 clip 只采样 1 个样本
```

或使用快速测试模式:

```bash
python scripts/train.py \
  --config configs/train_config.yaml \
  --max-batches 50
```

### Q3: 如何恢复中断的训练？

```bash
python scripts/train.py \
  --config configs/train_config.yaml \
  --auto-resume
```

恢复内容包括: 模型参数、优化器、调度器、GradScaler、最佳 val_loss

### Q4: 模型过拟合（val loss 上升，train loss 下降）

**可能原因及解决方案**：

| 原因 | 解决方案 |
|------|---------|
| 样本高度相关（滑动窗口） | 启用 `clip_level_sampling: true`（最重要！） |
| 学习率过高 | 降低至 `1e-4` 或更低 |
| 正则化不足 | 增加 `weight_decay: 1e-2` |
| 数据增强不足 | 确保 `enable_augmentation: true`（默认启用） |
| 每 clip 采样过多 | 减少 `samples_per_clip`（但不要低于 2） |
| BatchNorm 不稳定 | 已改用 GroupNorm |

**诊断方法**：
- 查看 TensorBoard 中的 `diag/pred_heatmap_max`，如果持续 < 0.1 说明热力图坍缩
- 检查 `epoch/train_loss` vs `epoch/val_loss` 曲线，如果 gap 过大说明过拟合

### Q5: 训练集样本数减少很多，会不会欠拟合？

**不会**，因为：
1. Clip-level 采样每 epoch 重新随机采样，50 epochs 累计可覆盖大量不同样本
2. 数据增强（ColorJitter + GaussianNoise）进一步增加多样性
3. 相比"看很多高度相关的样本"，"看较少但多样化的样本"更有利于泛化

**如果担心欠拟合**：
- 可以增加 `samples_per_clip` 到 3-4
- 观察 train loss 是否能正常下降

---

## 评估（Evaluate）

> ⚠️ **运行前请确保激活 conda 环境**：
> ```bash
> conda activate models
> ```

评估脚本：`scripts/evaluate.py`  
必须提供 `--checkpoint`。

如果你不传 `--use-history/--use-future/--use-action`，脚本会默认全部评估。

```bash
cd HeatmapVLN

python scripts/evaluate.py \
  --config configs/train_config.yaml \
  --checkpoint /path/to/ckpt.pth \
  --split val \
  --save-vis \
  --num-vis 20
```

---

## 配置文件说明（configs/train_config.yaml）

当前 `configs/` 目录只有一个配置：`train_config.yaml`。

你通常需要优先检查并修改：

- `data.root`：数据集根目录
- `data.val_split`：验证 split（例如 `val_unseen`）
- `model.llm.model_path`：本地 Qwen3-VL 权重路径（默认 `models/qwen_3_vl`）
- `log.out_dir` / `log.tensorboard_dir`：输出与日志目录

---

## 常见问题（FAQ）

### 1) 报错：找不到 split 目录 / clips

检查：

- `data.root` 是否指向正确路径
- split 是否存在：训练固定用 `train`；验证用 `data.val_split`
- clip 目录是否以 `clip_` 开头（数据集枚举逻辑依赖该前缀）

### 2) 动作一直是 0 或者 `action_valid=0`

这通常意味着 clip 下缺失 `actions.npy`，或当前帧索引超出动作数组长度。
如果你要训练 action head，建议确保每个 clip 都有 `actions.npy`。

### 3) 深度缺失会怎样？

深度是可选的。缺失时不会做遮挡检测，热力图仍然会生成。

---

## 关键文件结构（重要部分）

下面只列 **跑通训练/评估/推理最关键** 的文件与目）：

```text
HeatmapVLN/
  configs/
    train_config.yaml          # 唯一配置：数据路径、训练阶段、损失、日志等

  scripts/                                   # 三个入口脚本（README 命令都以它们为准）
    train.py                   # 训练：四阶段 curriculum（history/future heatmap + action + stop）
    evaluate.py                               # 评估：history/future/action + 可视化
    inference.py                              # 推理：对 video 或 dataset clip 生成 heatmap/actions

  src/
    data/
      vln_sliding_window_dataset.py           # 数据集：VLNSlidingWindowDataset（读取 meta/poses/rgb/depth/actions）
      keyframe_selector.py                    # 关键帧选择器（空间感知采样）
      frame_sampler.py                        # 贪心最大覆盖算法（体素化 + 增量选择）

    models/
      pipeline.py                             # 核心模型：VLNPipeline + VLNPipelineConfig

      heatmap/
        diffusion_heatmap_head.py             # DiffusionHeatmapHead（history/future 双头）
        diffusion/                            # 扩散网络组件
          config.py                           # DiffusionHeatmapConfig（超参配置）
          unet2d.py                           # ConditionalUnet2D（噪声预测网络）
          image_encoder.py                    # MultiModalConditionEncoder（条件编码）

      action/
        diffusion_action_head.py              # DiffusionActionHead（连续动作 dx,dy）
        stop_head.py                          # StopPredictionHead（二分类 STOP）
        action_config.py                      # 动作 head 配置/超参
        diffusion/                            # 扩散网络细节/组件

      qwen3_vl/                               # Qwen3-VL 集成模块
        __init__.py                           # 模块导出和 PACKING_AVAILABLE 检查
        integration.py                        # Qwen3VLIntegration（视觉语言特征提取）
        sequence_packing.py                   # Sequence Packing 高效批量训练

    utils/
      loss.py                                 # 损失函数（SimplifiedHeatmapLoss 等）
      visualization.py                        # 可视化工具
```



# HeatmapVLN

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.1+-ee4c2c.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)
![CUDA](https://img.shields.io/badge/CUDA-12.0+-76b900.svg)

**基于 Qwen3-VL 的视觉语言导航热力图与轨迹预测系统**

[快速开始](#快速开始) •
[模型架构](#模型架构) •
[训练指南](#训练) •
[配置说明](#配置说明) •
[常见问题](#常见问题)

</div>

---

## 📖 概述

HeatmapVLN 是一个用于视觉语言导航（VLN）任务的深度学习框架。通过历史视频帧、当前观测和自然语言指令，生成空间热力图和连续动作预测，为导航提供关键位置信息。

<p align="center">
  <img src="assets/architecture.png" width="800" alt="Architecture">
</p>

### ✨ 核心特性

| 特性 | 描述 |
|:-----|:-----|
| 🔥 **历史热力图** | 基于 Diffusion 生成历史相机位置在当前视角的热力图投影 |
| 🔮 **未来热力图** | 基于 Diffusion 预测未来位置在当前视角的投影（可选） |
| 👁️ **多视图支持** | 360° 全景图：4个方向同时预测，支持 circular_padding |
| 🎯 **轨迹预测** | 24 步连续轨迹预测 (x, y, θ)，Transformer + Diffusion |
| 📊 **进度估计** | 任务完成进度回归 (0-1)，3层 MLP |
| 🛑 **停止预测** | 基于 Focal Loss 的二分类（已弃用，用 Progress 替代） |
| 👁️ **可见性预测** | 预测目标位置是否在当前视角可见，控制假阳性 |
| 🧠 **Qwen3-VL 骨干** | 视觉语言预训练模型，支持可选 LoRA 微调 |
| 📡 **Multi-Layer Features** | 从 LLM 多层提取特征并融合 (CVPR 2025 最佳实践) |
| ⚡ **Sequence Packing** | 高效批量训练，消除 padding 浪费 |
| 🎛️ **模块化设计** | 可独立启用/禁用各预测头 |
| 🔄 **360° 全景支持** | circular_padding 处理左右边界连续性 |

---

## 📋 目录

- [快速开始](#快速开始)
  - [环境要求](#环境要求)
  - [安装](#安装)
  - [模型准备](#模型准备)
- [使用指南](#使用指南)
  - [训练](#训练)
  - [推理](#推理)
  - [评估](#评估)
- [模型架构](#模型架构)
  - [整体架构](#整体架构)
  - [热力图生成模块](#热力图生成模块)
  - [轨迹预测模块](#轨迹预测模块)
- [数据集](#数据集)
  - [数据格式](#数据格式)
  - [采样策略](#采样策略)
- [配置说明](#配置说明)
- [常见问题](#常见问题)
- [致谢](#致谢)
- [许可证](#许可证)

---

## 🚀 快速开始

### 环境要求

- Python 3.12+
- PyTorch 2.1+
- CUDA 12.0+
- 40GB+ GPU 显存（推荐 A100/H100）

### 安装

**方式一：Docker 部署（推荐）**

```bash
./docker/docker-run.sh
```

详细说明请参阅 [Docker 使用指南](docker/DOCKER.md)

**方式二：本地安装**

```bash
# 创建 conda 环境
conda create -n heatmapvln python=3.12 -y
conda activate heatmapvln

# 安装依赖
pip install -r requirements.txt

# 可选：安装 FlashAttention 2（推荐）
pip install flash-attn --no-build-isolation
```

### 模型准备

下载 Qwen3-VL 模型权重：

```bash
# 从 HuggingFace 下载
huggingface-cli download Qwen/Qwen3-VL-8B --local-dir models/qwen_3_vl

# 或从 ModelScope 下载
modelscope download Qwen/Qwen3-VL-8B --local_dir models/qwen_3_vl
```

### 快速验证

```bash
# 验证安装
python scripts/train.py --config configs/train_config.yaml --dry-run

# 快速训练测试
python scripts/train.py --config configs/train_config.yaml --epochs 2 --max-batches 5
```

---

## 📚 使用指南

### 训练

```bash
# 基础训练
python scripts/train.py --config configs/train_config.yaml

# 断点续训
python scripts/train.py --config configs/train_config.yaml --auto-resume
```

**常用参数：**

| 参数 | 说明 | 示例 |
|:-----|:-----|:-----|
| `--config` | 配置文件路径 | `configs/train_config.yaml` |
| `--resume` | 从指定检查点恢复 | `--resume ckpts/e005.pth` |
| `--auto-resume` | 自动从最新检查点恢复 | |
| `--dry-run` | 仅构建模型，不训练 | |
| `--epochs` | 训练轮数 | `--epochs 10` |

**后台训练：**

```bash
# 使用 tmux（推荐）
tmux new -s train
python scripts/train.py --config configs/train_config.yaml
# Ctrl+B D 退出，tmux attach -t train 重新进入

# 使用 nohup
nohup python -u scripts/train.py --config configs/train_config.yaml > train.log 2>&1 &
```

**TensorBoard 监控：**

```bash
tensorboard --logdir=/root/tf-logs/latest --port=6006
```

<details>
<summary>📊 TensorBoard 关键指标</summary>

| 分类 | 指标 | 说明 |
|:-----|:-----|:-----|
| **训练损失** | `train/loss` | 总损失 |
| | `train/history_heatmap_loss` | 历史热力图损失 |
| | `train/trajectory_loss` | 轨迹损失 (Transformer Diffusion) |
| | `train/progress_loss` | 进度损失 |
| **Multi-Layer Fusion** | `diag/fusion_weight_layer{i}` | 各层融合权重 |
| **热力图损失** | `diag/heatmap_diffusion_loss` | Min-SNR 加权 epsilon MSE 损失 |
| | `diag/heatmap_eps_mse_high_snr` | 低噪声区 (SNR>5) epsilon MSE |
| | `diag/heatmap_eps_mse_mid_snr` | 中噪声区 (0.5≤SNR≤5) epsilon MSE |
| | `diag/heatmap_eps_mse_low_snr` | 高噪声区 (SNR<0.5) epsilon MSE |
| **热力图诊断** | `diag/pred_heatmap_max` | 预测最大值（<0.1 可能坍缩） |
| | `diag/pred_heatmap_mean` | 预测均值 |
| | `diag/pred_heatmap_std` | 预测标准差 |
| | `diag/pred_heatmap_nonzero_ratio` | 非零像素比例 |
| | `diag/noise_std` | 真实噪声标准差 |
| | `diag/noise_pred_std` | 预测噪声标准差（应与 noise_std 接近） |
| **轨迹诊断** | `diag/trajectory_ade` | 平均位移误差 |
| | `diag/trajectory_fde` | 终点位移误差 |
| **进度诊断** | `diag/progress_mae` | 进度 MAE |
| | `diag/progress_pred_mean` | 预测进度均值 |
| | `diag/progress_gt_mean` | 真实进度均值 |
| | `diag/progress_boundary_error` | 进度边界误差 (0/1附件) |
| **资源监控** | `diag/gpu_memory_gb` | GPU 显存使用量 |
| | `diag/gpu_memory_reserved_gb` | GPU 预留显存 |

</details>

### 推理

```bash
# 对数据集 clip 推理
python scripts/inference.py \
  --clip /path/to/clip \
  --config configs/train_config.yaml \
  --checkpoint /path/to/best.pth \
  --output-dir ./outputs

# 对视频文件推理
python scripts/inference.py \
  --video /path/to/video.mp4 \
  --instruction "沿走廊前进并在门口右转" \
  --checkpoint /path/to/best.pth
```

### 评估

```bash
python scripts/evaluate.py \
  --config configs/train_config.yaml \
  --checkpoint /path/to/best.pth \
  --split val_unseen \
  --save-vis
```

**评估指标：**

| 类别 | 指标 | 说明 |
|:-----|:-----|:-----|
| **热力图** | Peak Error | 峰值位置误差（像素） |
| | IoU@0.1/0.3/0.5 | 多阈值交并比 |
| **轨迹** | ADE | 平均位移误差 |
| | FDE | 终点位移误差 |
| **进度** | MAE | 平均绝对误差 |
| **停止预测** | Accuracy/F1 | 分类准确率与 F1 分数 |

---

## 🏗️ 模型架构

### 整体架构

```
┌─────────────────────────────────────────────────────────────────────────────────────┐
│                              VLN Pipeline (Qwen3-VL)                               │
├─────────────────────────────────────────────────────────────────────────────────────┤
│                                                                                     │
│  输入                                                                                │
│  ┌────────────────┐  ┌────────────────┐  ┌────────────────┐                     │
│  │ History [K帧]  │  │ Current Frame  │  │  Instruction   │                     │
│  │  (多视图可选)   │  │   (224×224)    │  │    (文本)       │                     │
│  └───────┬────────┘  └───────┬────────┘  └───────┬────────┘                     │
│          └───────────────────┬┴───────────────────┘                                │
│                              │                                                     │
│                              ▼                                                     │
│                    ┌──────────────────┐                                           │
│                    │    Qwen3-VL      │  ← 冻结 (可选 LoRA 微调)                   │
│                    │   (Vision+LLM)   │     支持 Multi-Layer Features              │
│                    └────────┬─────────┘                                           │
│                             │                                                      │
│              hidden_states [B, seq, 4096]                                          │
│                             │                                                      │
│         ┌────────────────────┼────────────────────┐                                │
│         ▼                    ▼                    ▼                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                          │
│  │ Multi-Layer │    │    LLM      │    │    LLM      │                          │
│  │   Fusion    │    │  Projector  │    │  Projector  │                          │
│  │ (可选)       │    │ 4096→1024   │    │ (Vision)    │                          │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                          │
│         │                   │                   │                                  │
│         └───────────────────┼───────────────────┘                                  │
│                             │                                                      │
│                     llm_tokens [B, seq, 1024]                                      │
│                             │                                                      │
│         ┌────────────────────┼────────────────────┐                                │
│         ▼                    ▼                    ▼                                │
│  ┌─────────────┐    ┌─────────────┐    ┌─────────────┐                          │
│  │   Heatmap   │    │ Trajectory  │    │  Progress   │                          │
│  │    Head     │    │    Head     │    │    Head     │                          │
│  │  (Diffusion)│    │ (Transformer│    │    (MLP)    │                          │
│  │  +Visible   │    │   +Diffusion)│    │             │                          │
│  └──────┬──────┘    └──────┬──────┘    └──────┬──────┘                          │
│         │                 │                 │                                   │
│         ▼                 ▼                 ▼                                   │
│    Heatmap          Trajectory          Progress                                  │
│    [64×64]         [24, 3] (x,y,θ)      [0, 1]                                   │
│    + Visibility                                                                    │
│                                                                                     │
└─────────────────────────────────────────────────────────────────────────────────────┘
```

**训练模式：**
- **标准模式**: 逐样本处理
- **Sequence Packing**: 多样本打包成单一长序列 (FlashAttention Varlen)

**预测头输出：**
| 预测头 | 输出维度 | 方法 |
|:-------|:---------|:-----|
| History Heatmap | [B, 64, 64] | Conditional UNet2D + DDPM |
| Future Heatmap | [B, 64, 64] | Conditional UNet2D + DDPM |
| Visibility | [B, 1] | 3层MLP (可选) |
| Trajectory | [B, 24, 3] | Transformer Decoder + DDPM |
| Progress | [B, 1] | 3层MLP |

### 热力图生成模块

基于条件扩散模型（Diffusion）生成 64×64 空间热力图，标记历史相机位置在当前观测中的投影分布。支持多种热力图：
- **历史热力图** (History Heatmap)：历史相机位置在当前视角的投影
- **未来热力图** (Future Heatmap)：预测的未来位置在当前视角的投影（可选）
- **多视图热力图** (Multi-view)：4个方向的全景视图热力图

采用**双路径条件注入**架构，解决将整个 LLM token 序列压缩为单向量的信息瓶颈：

**架构：**

```
LLM Tokens (B, ~900, 1024)
    │
    ├──→ AttentionPooling ──→ global_cond (B, 1024) ──→ FiLM (所有 ResBlock)
    │                                                      ↓
    ├──→ LinearProj ──→ seq_cond (B, ~900, 1024) ──→ Cross-Attention
    │                                                      ↓
Current Frame(s) → CNN Encoder (ResNet-18/轻量CNN) ──→ img_cond ──→ Fusion
    │                                                                 ↓
    │                                                     ConditionalUnet2D
    │                                                                 ↓
    │                                                    DDPM + CFG → Heatmap
    │                                                                 ↓
    └──→ [可选] Visibility Head → 3层MLP → Visibility Score (是否可见)
```

**关键技术：**

| 技术 | 说明 |
|:-----|:-----|
| **双路径条件注入** | FiLM (全局向量) + 序列 Cross-Attention (保留完整 LLM token 序列) |
| **序列 Cross-Attention** | UNet 各空间位置 attend 到完整 LLM 序列，避免信息瓶颈 |
| **Image Encoder** | CNN (ResNet-18 或轻量 CNN) 编码当前图像，提供像素级空间特征 |
| **空间特征注入** | CNN 多尺度特征注入 UNet skip connections，解决全局池化导致的空间信息丢失 |
| **Multi-View 支持** | 360° 全景图：4个方向同时预测，支持 circular_padding 处理边界连续性 |
| **Multi-Layer Features** | 从 LLM 多层提取特征并融合 (CVPR 2025 最佳实践) |
| **Classifier-Free Guidance** | 训练时随机丢弃条件，推理时增强引导 (scale=2.0-4.0) |
| **Focal Loss** | 70% 标准 MSE + 30% 峰值加权，关注关键区域 |
| **x0 重构损失** | 直接监督输出质量，补充 epsilon loss |
| **Dice Loss** | treats sparse signal correctly，无背景梯度浪费 |
| **稀疏性正则化** | L1 正则化，鼓励大部分像素为 0 |
| **峰值距离损失** | 可微分 Soft-Argmax + NMS，多峰感知优化峰值位置准确度 |
| **负样本零目标损失** | SNR 门控：只对高质量样本（低时间步）施加零约束 |
| **可见性预测头** | 3层MLP预测目标是否可见，消除假阳性 |
| **空间 Softmax 锐化** | 推理后处理，温度 0.1 集中能量到峰值区域 |
| **LoRA 微调 (可选)** | 对 Qwen3-VL 最后 N 层加 LoRA，增强空间推理能力 |
| **轨迹增强** | 训练时随机旋转/缩放轨迹，提升泛化能力 |
| **FGR2R 子指令** | 支持动态子指令，适应子序列采样 |
| **Sequence Packing** | 批量训练优化，多样本打包成长序列 |

### 轨迹预测模块

基于 Transformer Decoder + Diffusion 的 24 步轨迹预测。

| 组件 | 输出 | 说明 |
|:-----|:-----|:-----|
| `TransformerActionHead` | (x, y, θ) × 24 | Transformer Decoder + DDPM (推荐) |
| `DiffusionActionHead` | (x, y, θ) × 24 | UNet1D + DDPM (legacy) |
| `ProgressPredictionHead` | [0, 1] | 3 层 MLP 回归 |
| `StopHead` | binary | 基于 Focal Loss 的二分类，判断是否到达目标 |

---

## 📂 数据集

### 数据格式

```
<data_root>/
├── train/
│   └── <scene_id>/
│       └── clip_000000/
│           ├── meta.json          # 元信息
│           ├── poses.json         # T 个 4×4 位姿矩阵
│           ├── rgb/               # RGB 图像序列
│           │   ├── 000000.png
│           │   └── ...
│           ├── actions.npy        # 连续动作 [T, 2] (agent-local 2D 位移 dx, dy)
│           └── discrete_actions.npy  # 离散动作 [T] (前进/左转/右转/停止)
└── val_unseen/
    └── ...
```

### 采样策略

| 模式 | 数据多样性 | 说明 |
|:-----|:----------:|:-----|
| 滑动窗口 | ⭐ | 固定步长遍历 |
| Clip-level | ⭐⭐ | 每 clip 随机采样 |
| **随机子序列** | ⭐⭐⭐ | 动态子序列 + 子指令（推荐） |

<details>
<summary>📖 随机子序列采样详解</summary>

每 epoch 从同一 clip 生成不同的子序列，大幅增加数据多样性：

```
原始 Clip: [帧0, 帧1, ..., 帧99]

子序列1: [10, 50]  → progress: 0% → 100%
子序列2: [30, 80]  → progress: 0% → 100%
子序列3: [5, 70]   → progress: 0% → 100%
```

**配置：**

```yaml
data:
  trajectory:
    random_subsequence: true
    min_subsequence_length: 30
    subsequence_samples_per_clip: 5
    samples_per_clip: 30
    use_subinstruction: true     # 启用 FGR2R 子指令
    enable_trajectory_augmentation: true  # 轨迹增强（旋转/缩放）
```

**数据量计算：**

```
每 epoch = clips × subseq × samples = 1000 × 5 × 30 = 150,000 样本
```

</details>

---

## ⚙️ 配置说明

主配置文件：`configs/train_config.yaml`（通用配置）或 `configs/train_heatmap_config.yaml`（热力图专用配置）

<details>
<summary>📋 完整配置示例</summary>

```yaml
# 模型配置
model:
  llm:
    model_path: ./models/qwen_3_vl
    attn_implementation: flash_attention_2
    enable_packing: true          # Sequence Packing 高效训练
    max_seq_length: 8192

    # LoRA 微调（可选，默认关闭）
    use_lora: false
    lora_rank: 16
    lora_alpha: 32
    lora_num_layers: 4            # 最后 4 层

  # 热力图头配置
  heatmap_head:
    enable_history: true           # 生成历史位置热力图
    enable_future: false          # 生成未来位置热力图
    cond_dim: 1024
    block_out_channels: [128, 256, 512, 512]
    attention_levels: [1, 2, 3]
    num_inference_steps: 50
    cfg_drop_prob: 0.15
    cfg_scale: 2.0

    # 双路径条件注入
    use_image_encoder: true             # CNN 编码当前图像
    use_spatial_injection: true         # CNN 多尺度特征注入 skip connections
    use_sequence_conditioning: true     # 序列级 Cross-Attention
    seq_cross_attn_heads: 8
    seq_cross_attn_head_dim: 64

    # 可见性预测头
    use_visibility_head: true
    visibility_loss_weight: 1.0
    visibility_threshold: 0.7

    # 推理后处理
    sharpen_temperature: 0.1            # 空间 Softmax 锐化

    # 损失函数权重
    x0_loss_weight: 1.0                  # x0 重构损失
    sparsity_loss_weight: 0.5           # L1 稀疏性正则化
    peak_distance_loss_weight: 2.0      # 可微分峰值距离损失
    negative_sample_weight: 0.3         # 负样本权重

  # 动作头配置
  action_head:
    enable: true
    type: transformer                   # transformer (推荐) 或 legacy

    # 停止预测头
    stop_head:
      enable: true

  # 进度预测头
  progress_head:
    enable: true

# 优化器配置
optim:
  batch_size: 16
  grad_accum_steps: 2             # 有效 batch = 32
  heatmap_lr: 1.0e-4
  llm_projector_lr: 5.0e-5        # 投影层使用更低 LR
  lora_lr: 1.0e-5                 # LoRA 使用极低 LR
```

</details>

---

## ❓ 常见问题

<details>
<summary><b>显存不足 (CUDA OOM)</b></summary>

减小 batch size 并增加梯度累积：

```yaml
optim:
  batch_size: 2
  grad_accum_steps: 16  # 有效 batch = 32
```

</details>

<details>
<summary><b>热力图全黑</b></summary>

检查 TensorBoard 中 `diag/pred_heatmap_max`：
- 如果 < 0.1，说明热力图坍缩
- 确保 `use_image_encoder: true` 和 `use_sequence_conditioning: true` 已启用
- 检查 `cfg_scale` 是否过高（推荐 2.0-4.0）

</details>

<details>
<summary><b>如何恢复训练？</b></summary>

```bash
python scripts/train.py --config configs/train_config.yaml --auto-resume
```

</details>

<details>
<summary><b>如何启用/禁用特定预测头？</b></summary>

```yaml
# 热力图头
model:
  heatmap_head:
    enable_history: true    # 历史热力图
    enable_future: false    # 未来热力图

# 动作头
model:
  action_head:
    enable: true
    type: transformer      # transformer (推荐) 或 legacy

# 停止预测头 (legacy，已弃用)
model:
  stop_head:
    enable: true

# 进度预测头
model:
  progress_head:
    enable: true
```

</details>

<details>
<summary><b>Diffusion 训练/推理步数如何配置？</b></summary>

推荐配置（4:1 比例）：
```yaml
model:
  heatmap_head:
    num_train_timesteps: 200    # 训练步数
    num_inference_steps: 50     # 推理步数 (训练步数的 1/4)
```

</details>

<details>
<summary><b>如何配置 LoRA 微调？</b></summary>

```yaml
model:
  llm:
    use_lora: true
    lora_rank: 16
    lora_alpha: 32
    lora_num_layers: 4      # 最后 4 层

optim:
  lora_lr: 1.0e-5          # LoRA 使用极低学习率
```

</details>

---

## 📁 项目结构

```
HeatmapVLN/
├── configs/
│   ├── train_config.yaml           # 通用训练配置
│   └── train_heatmap_config.yaml  # 热力图训练配置
├── scripts/
│   ├── train.py                    # 训练脚本
│   ├── evaluate.py                 # 评估脚本
│   ├── inference.py                # 推理脚本
│   ├── eval_heatmap.py             # 热力图专用评估脚本
│   ├── visualize_heatmap.py       # 热力图可视化脚本
│   └── visualize_trajectory_heatmaps.py  # 轨迹热力图可视化
├── src/
│   ├── data/                       # 数据加载
│   │   ├── vln_sliding_window_dataset.py  # 滑动窗口 + 轨迹数据集
│   │   ├── tokenized_dataset.py    # Qwen3-VL tokenization 数据集
│   │   └── packing_collator.py     # Sequence Packing collator
│   ├── models/                     # 模型定义
│   │   ├── pipeline.py             # VLNPipeline 主模块
│   │   ├── qwen3_vl/               # Qwen3-VL 集成 (含 LoRA 支持)
│   │   ├── heatmap/                # 热力图模块
│   │   │   ├── diffusion_heatmap_head.py  # Diffusion 热力图头
│   │   │   └── diffusion/
│   │   │       ├── config.py       # 配置 (含序列 Cross-Attention)
│   │   │       ├── unet2d.py       # UNet2D (双路径条件注入)
│   │   │       ├── image_encoder.py # 条件编码器 (ResNet-18)
│   │   │       └── positional_embedding.py
│   │   └── action/                 # 动作模块
│   │       ├── transformer_action_head.py  # Transformer DDPM (推荐)
│   │       ├── diffusion_action_head.py    # UNet1D DDPM (legacy)
│   │       ├── progress_head.py     # 进度预测头
│   │       ├── stop_head.py         # 停止预测头
│   │       └── utils.py
│   └── utils/                      # 工具函数
│       ├── gpu_heatmap.py           # GPU 热力图计算
│       ├── loss.py                  # 损失函数 (Focal, SNR-gated)
│       ├── notifier.py              # 飞书通知
│       ├── visualization.py         # 可视化工具
│       └── frame_vis_utils.py       # 帧可视化工具
├── docker/                         # Docker 配置
├── assets/                        # 资源文件
│   └── architecture.png            # 架构图
├── requirements.txt
└── README.md
```

---

## 🙏 致谢

- [Qwen3-VL](https://github.com/QwenLM/Qwen-VL) - 视觉语言骨干模型
- [InternNav](https://github.com/OpenRobotLab/InternNav) - Transformer Action Head 参考实现
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) - 扩散策略参考

---

## 📄 许可证

本项目采用 [MIT License](LICENSE) 开源许可证。

---

<div align="center">

**如果这个项目对你有帮助，请给一个 ⭐ Star！**

</div>

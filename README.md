# HeatmapVLN

<div align="center">

![Python](https://img.shields.io/badge/Python-3.12-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-ee4c2c.svg)
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
| 🔥 **热力图生成** | 基于条件扩散模型生成历史位置热力图 |
| 🎯 **轨迹预测** | 24 步连续轨迹预测 (x, y, θ) |
| 📊 **进度估计** | 任务完成进度回归 (0-1) |
| 🧠 **Qwen3-VL 骨干** | 利用强大的视觉语言预训练模型 |
| ⚡ **Sequence Packing** | 高效批量训练，消除 padding 浪费 |
| 🎛️ **模块化设计** | 可独立启用/禁用各预测头 |

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
| | `train/heatmap_loss` | 热力图损失 |
| | `train/trajectory_loss` | 轨迹损失 |
| | `train/progress_loss` | 进度损失 |
| **热力图诊断** | `diag/pred_heatmap_max` | 预测最大值（<0.1 可能坍缩） |
| | `diag/heatmap_focal_ratio` | focal/base 比值 |
| | `diag/heatmap_regional_ratio` | 区域损失比值 |
| **区域损失细化** | `diag/hm_loss_center` | 中心区域(前方)损失 |
| | `diag/hm_loss_left` | 左侧区域(后方左)损失 |
| | `diag/hm_loss_right` | 右侧区域(后方右)损失 |
| | `diag/hm_loss_top` | 上部区域损失 |
| | `diag/hm_loss_bottom` | 下部区域损失 |
| **热力图质量** | `diag/hm_peak_distance` | Peak 位置误差(像素) |
| | `diag/hm_peak_dx` | Peak X 方向误差 |
| | `diag/hm_peak_dy` | Peak Y 方向误差 |
| | `diag/hm_peak_iou` | Peak 区域 IoU |
| | `diag/hm_peak_conf_ratio` | 峰值置信度比值(pred/gt) |
| **轨迹诊断** | `diag/trajectory_ade` | 平均位移误差 |
| | `diag/trajectory_fde` | 终点位移误差 |

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
| | IoU | 阈值交并比 |
| **轨迹** | ADE | 平均位移误差 |
| | FDE | 终点位移误差 |
| **进度** | MAE | 平均绝对误差 |

---

## 🏗️ 模型架构

### 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         VLN Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入                                                           │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ History [K帧]│  │ Current Frame│  │ Instruction  │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         └─────────────────┴─────────────────┘                   │
│                           │                                     │
│                           ▼                                     │
│                  ┌─────────────────┐                            │
│                  │    Qwen3-VL     │  ← 参数冻结                │
│                  │   (Backbone)    │                            │
│                  └────────┬────────┘                            │
│                           │                                     │
│              hidden_states [B, seq, 2048]                       │
│                           │                                     │
│         ┌─────────────────┼─────────────────┐                   │
│         ▼                 ▼                 ▼                   │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │   Heatmap   │   │ Trajectory  │   │  Progress   │           │
│  │    Head     │   │    Head     │   │    Head     │           │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘           │
│         │                 │                 │                   │
│         ▼                 ▼                 ▼                   │
│    Heatmap          Trajectory          Progress                │
│    [64×64]         [24, 3]              [0, 1]                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### 热力图生成模块

基于条件扩散模型（Diffusion）生成 64×64 空间热力图，标记历史位置在当前视野中的分布。

**架构：**

```
LLM Tokens → Attention Pooling → Condition Encoder
                                        ↓
ConditionalUnet2D (Cross-Attention + FiLM) → DDPM + CFG → Heatmap
```

**关键技术：**

| 技术 | 说明 |
|:-----|:-----|
| **Cross-Attention** | UNet 中间层添加交叉注意力，增强条件注入 |
| **Classifier-Free Guidance** | 训练时随机丢弃条件，推理时增强引导 |
| **Focal Loss** | 70% 标准 MSE + 30% 峰值加权 |
| **区域感知损失** | 对稀疏区域（前方、上下）增加权重 |
| **360° Circular Padding** | 支持全景图水平边界连续性 |

<details>
<summary>📐 区域感知损失详解</summary>

针对 R2R 数据集热力图分布不均问题（89% 集中在垂直中间，80% 集中在后方），对稀疏区域给予更高权重：

```
全景图 64×64 热力图区域权重:

         ┌───────────────────────────────────────┐
   上    │  ×1.5   │  ×3.0 (1.5×2.0)   │  ×1.5   │
(稀疏)   │  左后   │     中心/前方     │  右后   │
         ├─────────┼───────────────────┼─────────┤
   中    │  ×1.0   │       ×2.0        │  ×1.0   │  ← 89% 分布
(密集)   │ 后方左  │      正前方       │ 后方右  │
         ├─────────┼───────────────────┼─────────┤
   下    │  ×1.5   │  ×3.0 (1.5×2.0)   │  ×1.5   │
(稀疏)   │  左后   │     中心/前方     │  右后   │
         └───────────────────────────────────────┘
```

- **中心区域（前方）**：权重 ×2.0
- **上下区域**：权重 ×1.5
- **区域损失权重**：占总损失 20%

</details>

### 轨迹预测模块

基于 Transformer Decoder + Diffusion 的 24 步轨迹预测。

| 组件 | 输出 | 说明 |
|:-----|:-----|:-----|
| `TransformerActionHead` | (x, y, θ) × 24 | Transformer Decoder + DDPM |
| `ProgressPredictionHead` | [0, 1] | 3 层 MLP 回归 |

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
│           └── actions.npy        # 连续动作 [T, 2]
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
```

**数据量计算：**

```
每 epoch = clips × subseq × samples = 1000 × 5 × 30 = 150,000 样本
```

</details>

---

## ⚙️ 配置说明

主配置文件：`configs/train_config.yaml`

<details>
<summary>📋 完整配置示例</summary>

```yaml
# 数据配置
data:
  root: dataset_with_actions
  val_split: val_unseen
  trajectory:
    random_subsequence: true
    min_subsequence_length: 30
    subsequence_samples_per_clip: 5

# 模型配置
model:
  llm:
    model_path: ./models/qwen_3_vl
    attn_implementation: flash_attention_2
  
  heatmap_head:
    enable_history: true
    cond_dim: 1024
    block_out_channels: [128, 256, 512, 512]
    attention_levels: [2, 3]
    num_inference_steps: 20
    cfg_drop_prob: 0.1
    cfg_scale: 3.0
    regional_loss_enabled: true
    regional_center_alpha: 2.0
    regional_vertical_alpha: 1.5
    regional_loss_weight: 0.2

# 优化器配置
optim:
  batch_size: 32
  grad_accum_steps: 4
  heatmap_lr: 1.0e-4
  action_lr: 1.0e-4
  progress_lr: 1.0e-4

# 损失权重
loss:
  history_weight: 1.0
  trajectory_weight: 1.0
  progress_weight: 1.0
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
- 确保使用 `use_image_encoder: false`（LLM-only 模式）

</details>

<details>
<summary><b>如何恢复训练？</b></summary>

```bash
python scripts/train.py --config configs/train_config.yaml --auto-resume
```

</details>

---

## 📁 项目结构

```
HeatmapVLN/
├── configs/
│   └── train_config.yaml           # 训练配置
├── scripts/
│   ├── train.py                    # 训练脚本
│   ├── evaluate.py                 # 评估脚本
│   └── inference.py                # 推理脚本
├── src/
│   ├── data/                       # 数据加载
│   │   ├── vln_sliding_window_dataset.py
│   │   └── tokenized_dataset.py
│   └── models/                     # 模型定义
│       ├── pipeline.py             # VLNPipeline 主模块
│       ├── qwen3_vl/               # Qwen3-VL 集成
│       ├── heatmap/                # 热力图模块
│       └── action/                 # 动作模块
├── docker/                         # Docker 配置
├── docs/                           # 文档
├── requirements.txt
└── README.md
```

### 训练输出结构

每次训练创建独立目录，便于管理和对比：

```
/root/autodl-tmp/vln_training_outputs/
├── run_20260123_001234/            # 训练 1
│   ├── ckpts/                      # 检查点
│   │   ├── epoch_001.pth
│   │   ├── best.pth
│   │   └── latest.pth
│   ├── vis/                        # 可视化
│   │   ├── train/                  # 训练热力图
│   │   └── val/                    # 验证热力图
│   ├── plots/
│   │   ├── curves.png              # 训练曲线
│   │   └── history.json
│   └── train.log
├── run_20260123_120000/            # 训练 2
│   └── ...
└── latest → run_20260123_120000    # 符号链接指向最新
```

**断点续训**会自动继续使用之前的目录：
```bash
python scripts/train.py --config configs/train_config.yaml --auto-resume
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

# HeatmapVLN

基于 **Qwen3-VL** 视觉语言模型的视觉语言导航（VLN）热力图与动作预测系统。

<img src="assets/architecture.png" width="800">

*通过 N 帧历史视频序列、当前观测和导航指令，生成空间热力图和连续/离散动作预测，为导航提供关键位置信息。*

---

## 目录

- [功能特性](#功能特性)
- [快速开始](#快速开始)
  - [环境安装](#环境安装)
  - [模型准备](#模型准备)
  - [快速验证](#快速验证)
- [使用指南](#使用指南)
  - [训练](#训练)
  - [推理](#推理)
  - [评估](#评估)
- [数据集准备](#数据集准备)
- [配置说明](#配置说明)
- [模型架构](#模型架构)
  - [整体架构](#整体架构)
  - [Qwen3-VL 集成模块](#qwen3-vl-集成模块)
  - [热力图生成模块](#热力图生成模块diffusionheatmaphead)
  - [动作预测模块](#动作预测模块)
- [常见问题](#常见问题-faq)
- [项目结构](#项目结构)

---

## 功能特性

| 功能 | 描述 |
|------|------|
| **🔥 热力图生成** | 基于扩散模型生成历史位置热力图，标记 agent 来时的方向 |
| **🎯 连续动作预测** | 2D 连续位移 (dx, dy) 或多步轨迹 (x, y, θ) × T |
| **🛑 停止预测** | 二分类判断是否执行 STOP，或连续进度值 (0-1) |
| **🧠 Qwen3-VL 骨干** | 使用 Qwen3-VL 视觉语言模型提取多模态特征 |
| **⚡ Sequence Packing** | 基于官方 fine-tuning 框架的高效批量训练 |
| **📊 TensorBoard 监控** | 实时可视化训练曲线和热力图 |

---

## 快速开始

### 环境安装

```bash
cd HeatmapVLN

# 创建 conda 环境（Python 3.11+）
conda create -n models python=3.11 -y
conda activate models

# 安装依赖
pip install -U pip
pip install -r requirements.txt

# 可选：安装 FlashAttention 2（推荐，显著提升性能）
pip install flash-attn --no-build-isolation
```

> **CUDA 版本**：如需特定 CUDA 版本的 PyTorch，请参考 `requirements.txt` 中的注释。

### 模型准备

下载 Qwen3-VL 模型权重并放置到 `models/qwen_3_vl/` 目录：

```bash
# 方式 1：从 HuggingFace 下载
huggingface-cli download Qwen/Qwen3-VL-2B --local-dir models/qwen_3_vl

# 方式 2：从 ModelScope 下载
modelscope download Qwen/Qwen3-VL-2B --local_dir models/qwen_3_vl
```

### 快速验证

```bash
# 验证安装（构建模型但不训练）
python scripts/train.py --config configs/train_config.yaml --dry-run

# 快速训练测试（2 个 epoch，每 epoch 5 个 batch）
python scripts/train.py --config configs/train_config.yaml --epochs 2 --max-batches 5
```

---

## 使用指南

### 训练

```bash
# 激活环境
conda activate models

# 开始训练
python scripts/train.py --config configs/train_config.yaml
```

**常用参数**：

| 参数 | 说明 | 示例 |
|------|------|------|
| `--config` | 配置文件路径 | `configs/train_config.yaml` |
| `--resume` | 从检查点恢复 | `--resume /path/to/ckpt.pth` |
| `--auto-resume` | 自动从最新检查点恢复 | |
| `--dry-run` | 只构建模型，不训练 | |
| `--max-batches` | 限制每 epoch 的 batch 数 | `--max-batches 50` |
| `--epochs` | 训练 epoch 数 | `--epochs 10` |

**后台训练**：

```bash
cd /root/HeatmapVLN && \
  source /root/miniconda3/etc/profile.d/conda.sh && \
  conda activate models && \
  nohup python -u scripts/train.py --config configs/train_config.yaml > train.log 2>&1 &

# 查看日志
tail -f train.log
```

**TensorBoard 监控**：

```bash
tensorboard --logdir=/root/tf-logs --port=6006
```

### 推理

推理脚本支持热力图、轨迹预测和进度预测：

```bash
# 对数据集 clip 推理（默认输出所有预测）
python scripts/inference.py \
  --clip dataset_with_actions/val_unseen/<scene_id>/clip_000000 \
  --config configs/train_config.yaml \
  --checkpoint /path/to/best_model.pth \
  --output-dir ./outputs_inference

# 对视频文件推理
python scripts/inference.py \
  --video /path/to/video.mp4 \
  --instruction "沿走廊前进并在门口右转" \
  --config configs/train_config.yaml \
  --checkpoint /path/to/best_model.pth \
  --output-dir ./outputs_inference

# 只输出特定预测
python scripts/inference.py \
  --clip /path/to/clip \
  --output-heatmap \
  --output-trajectory \
  --output-progress
```

**输出文件**：

| 文件 | 说明 |
|------|------|
| `*_combined.png` | 综合可视化（热力图 + 轨迹 + 进度） |
| `*_heatmap.png` | 历史位置热力图可视化 |
| `*_heatmap.npy` | 热力图原始数据 |
| `*_trajectory.png` | 24 步轨迹可视化 |
| `*_trajectory.npy` | 轨迹原始数据 [24, 3] (dx, dy, dyaw) |
| `*_trajectory.txt` | 轨迹文本格式 |
| `*_summary.yaml` | 推理摘要（进度、热力图最大值等） |

### 评估

评估脚本计算热力图、轨迹和进度的定量指标：

```bash
# 完整评估（热力图 + 轨迹 + 进度）
python scripts/evaluate.py \
  --config configs/train_config.yaml \
  --checkpoint /path/to/best_model.pth \
  --split val_unseen \
  --save-vis \
  --num-vis 20

# 只评估特定指标
python scripts/evaluate.py \
  --checkpoint /path/to/best_model.pth \
  --eval-heatmap \
  --eval-trajectory \
  --eval-progress

# 使用 Sequence Packing 加速
python scripts/evaluate.py \
  --checkpoint /path/to/best_model.pth \
  --use-packing
```

**评估指标**：

| 类别 | 指标 | 说明 |
|------|------|------|
| **热力图** | Peak Error | 峰值位置误差（像素） |
| | IoU | 阈值交并比 (threshold=0.3) |
| | Cosine Sim | 余弦相似度 |
| | MAE | 平均绝对误差 |
| **轨迹** | ADE | 平均位移误差 (Average Displacement Error) |
| | FDE | 最终位移误差 (Final Displacement Error) |
| **进度** | MAE | 平均绝对误差 |
| | Accuracy | 阈值准确率 (threshold=0.1) |
| | Boundary Acc | 边界准确率 (progress ≈ 0 或 1) |

**可视化输出**：

评估时使用 `--save-vis` 会保存到 `<out_dir>/eval_vis/`，每个样本包含：
- 当前帧、GT 热力图、预测热力图
- GT 轨迹 vs 预测轨迹对比
- 进度预测对比

---

## 数据集准备

### 目录结构

```
<data_root>/
├── train/
│   └── <scene_id>/
│       └── clip_000000/
│           ├── meta.json              # 必需：num_frames, instruction
│           ├── poses.json             # 必需：T 个 4×4 位姿矩阵
│           ├── rgb/                   # 必需：RGB 图像序列
│           │   ├── 000000.png
│           │   └── ...
│           ├── depth/                 # 可选：深度图
│           │   ├── 000000.npy
│           │   └── ...
│           ├── actions.npy            # 可选：连续动作 [T, 2]
│           └── discrete_actions.npy   # 可选：离散动作 [T]
└── val_unseen/
    └── ...
```

### 文件说明

| 文件 | 必需 | 格式 | 说明 |
|------|:----:|------|------|
| `meta.json` | ✅ | JSON | 至少包含 `num_frames`；可含 `instruction` |
| `poses.json` | ✅ | JSON | 长度为 T 的 4×4 位姿矩阵列表 |
| `rgb/*.png` | ✅ | PNG | 按 6 位零填充命名 |
| `depth/*.npy` | ❌ | NPY | 深度图，用于遮挡检测 |
| `actions.npy` | ❌ | NPY | 连续动作 [T, 2] (dx, dy) |
| `discrete_actions.npy` | ❌ | NPY | 离散动作 [T] (0-3) |

### 数据采样策略

为防止过拟合，训练集使用 **Clip-level 采样**：

```yaml
# configs/train_config.yaml
data:
  sliding_window:
    clip_level_sampling: true   # 启用 clip-level 采样
    samples_per_clip: 2         # 每 clip 每 epoch 采样 2 个
```

| 采样方式 | 单 epoch 样本数 | 样本相关性 | 推荐 |
|---------|---------------|-----------|:----:|
| 滑动窗口 (stride=1) | ~133,000 | 极高 | ❌ |
| **Clip-level (N=2)** | ~2,800 | 低 | ✅ |

---

## 配置说明

配置文件：`configs/train_config.yaml`

### 关键配置项

```yaml
# 数据
data:
  root: dataset_with_actions      # 数据集路径
  val_split: val_unseen           # 验证集 split

# 模型
model:
  llm:
    model_path: ./models/qwen_3_vl
    attn_implementation: flash_attention_2
  heatmap_head:
    enable_history: true
    use_image_encoder: false      # 推荐 false (LLM-only)
  action_head:
    enable: true
  stop_head:
    enable: true

# 优化器
optim:
  batch_size: 32
  grad_accum_steps: 4             # 有效 batch = 128
  heatmap_lr: 1.0e-4
  action_lr: 1.0e-4

# 损失权重
loss:
  history_weight: 1.0
  action_weight: 1.0
  stop_weight: 0.5

# 日志
log:
  out_dir: vln_training_outputs
  use_tensorboard: true
```

---

## 模型架构

### 整体架构

```
┌─────────────────────────────────────────────────────────────────┐
│                         VLN Pipeline                            │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  输入:                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐          │
│  │ History      │  │  Current     │  │ Instruction  │          │
│  │ Frames [K帧] │  │  Frame       │  │   Text       │          │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘          │
│         │                 │                 │                   │
│         └─────────────────┴─────────────────┘                   │
│                           │                                     │
│                           ▼                                     │
│                  ┌─────────────────┐                            │
│                  │   Qwen3-VL      │  ← 参数冻结                │
│                  │   (骨干网络)    │                            │
│                  └────────┬────────┘                            │
│                           │                                     │
│              hidden_states [B, seq, 2048]                       │
│                           │                                     │
│         ┌─────────────────┼─────────────────┐                   │
│         ▼                 ▼                 ▼                   │
│  ┌─────────────┐   ┌─────────────┐   ┌─────────────┐           │
│  │  Heatmap    │   │   Action    │   │    Stop     │           │
│  │    Head     │   │    Head     │   │    Head     │           │
│  └──────┬──────┘   └──────┬──────┘   └──────┬──────┘           │
│         │                 │                 │                   │
│         ▼                 ▼                 ▼                   │
│    Heatmap           Actions            Stop Prob               │
│   [64×64]            (dx,dy)             [0,1]                  │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

### Qwen3-VL 集成模块

使用 Qwen3-VL 作为视觉语言骨干网络，提取多模态特征。

**关键特性**：
- 参数冻结（不训练骨干）
- Flash Attention 2 加速
- 支持 Sequence Packing 高效训练

```python
from src.models.qwen3_vl import Qwen3VLIntegration, Qwen3VLConfig

config = Qwen3VLConfig(model_path="./models/qwen_3_vl")
qwen_vl = Qwen3VLIntegration(config)

outputs = qwen_vl(history_frames, current_frame, instruction)
hidden_states = outputs["hidden_states"]  # [B, seq, 2048]
```

**Sequence Packing**：基于官方 fine-tuning 框架，将多个样本打包成一个序列，消除 padding 浪费。

| 模式 | 显存利用率 | 说明 |
|------|-----------|------|
| 传统 Padding | ~50% | 大量 PAD token |
| **Sequence Packing** | ~100% | 无 PAD，最大化利用 |

<details>
<summary>📁 源代码位置</summary>

| 文件 | 说明 |
|------|------|
| `src/models/qwen3_vl/integration.py` | `Qwen3VLIntegration`, `Qwen3VLConfig` |
| `src/models/qwen3_vl/sequence_packing.py` | `FlattenedDataCollatorForVLN` |

</details>

---

### 热力图生成模块（DiffusionHeatmapHead）

使用条件扩散模型生成空间热力图。

**架构**：
```
LLM Tokens → Attention Pooling → Condition Encoder → ConditionalUnet2D → DDPM → Heatmap
```

**关键组件**：
| 组件 | 说明 |
|------|------|
| `AttentionPooling` | 可学习 query + 多头注意力聚合 |
| `MultiModalConditionEncoder` | LLM + Image 特征融合（推荐 LLM-only） |
| `ConditionalUnet2D` | FiLM 条件调制的 2D U-Net |
| `DDPMScheduler` | 100 步训练，10 步推理 |

**配置**：
```yaml
model:
  heatmap_head:
    enable_history: true
    cond_dim: 512
    use_image_encoder: false    # 推荐 LLM-only 模式
    llm_pool_method: attention
```

<details>
<summary>📁 源代码位置</summary>

| 文件 | 说明 |
|------|------|
| `src/models/heatmap/diffusion_heatmap_head.py` | `DiffusionHeatmapHead` |
| `src/models/heatmap/diffusion/unet2d.py` | `ConditionalUnet2D` |
| `src/models/heatmap/diffusion/image_encoder.py` | `MultiModalConditionEncoder` |

</details>

---

### 动作预测模块

提供多种动作生成方式：

| 组件 | 输出 | 说明 |
|------|------|------|
| `DiffusionActionHead` | (dx, dy) | 1D U-Net 扩散模型 |
| `TransformerActionHead` | (x,y,θ)×24 | Transformer Decoder + Diffusion |
| `StopPredictionHead` | STOP/继续 | 二分类器 + 混合 Focal Loss |
| `ProgressPredictionHead` | 0-1 | 任务进度回归 |

**配置**：
```yaml
model:
  action_head:
    enable: true
    action_dim: 2
    pred_horizon: 1
  stop_head:
    enable: true
    focal_gamma: 2.0
```

<details>
<summary>📁 源代码位置</summary>

| 文件 | 说明 |
|------|------|
| `src/models/action/diffusion_action_head.py` | `DiffusionActionHead` |
| `src/models/action/transformer_action_head.py` | `TransformerActionHead` |
| `src/models/action/stop_head.py` | `StopPredictionHead` |
| `src/models/action/progress_head.py` | `ProgressPredictionHead` |

</details>

---

## 常见问题 (FAQ)

<details>
<summary><b>Q1: 显存不足 (CUDA OOM)</b></summary>

**方案 1**：减小 batch size + 增加梯度累积
```yaml
optim:
  batch_size: 2
  grad_accum_steps: 16    # 有效 batch 保持不变
```

**方案 2**：减少每 clip 采样数
```yaml
data:
  sliding_window:
    samples_per_clip: 1
```

</details>

<details>
<summary><b>Q2: 如何恢复中断的训练？</b></summary>

```bash
python scripts/train.py --config configs/train_config.yaml --auto-resume
```

</details>

<details>
<summary><b>Q3: 热力图全黑怎么办？</b></summary>

检查 TensorBoard 中的 `diag/pred_heatmap_max`：
- 如果 < 0.1，说明热力图坍缩
- 确保 `use_image_encoder: false`（LLM-only 模式）
- 检查 GT 热力图是否正常

</details>

<details>
<summary><b>Q4: 模型过拟合（val loss 上升）</b></summary>

| 原因 | 解决方案 |
|------|---------|
| 滑动窗口采样 | 启用 `clip_level_sampling: true` |
| 学习率过高 | 降低至 `1e-4` |
| 正则化不足 | 增加 `weight_decay: 1e-2` |

</details>

<details>
<summary><b>Q5: 找不到数据集/clips</b></summary>

检查：
- `data.root` 是否指向正确路径
- clip 目录是否以 `clip_` 开头
- split 名称是否正确

</details>

---

## 项目结构

```
HeatmapVLN/
├── configs/
│   └── train_config.yaml           # 训练配置
├── scripts/
│   ├── train.py                    # 训练脚本
│   ├── evaluate.py                 # 评估脚本
│   └── inference.py                # 推理脚本
├── src/
│   ├── data/
│   │   ├── vln_sliding_window_dataset.py   # 数据集
│   │   └── tokenized_dataset.py            # Tokenization
│   ├── models/
│   │   ├── pipeline.py                     # VLNPipeline
│   │   ├── qwen3_vl/                       # Qwen3-VL 集成
│   │   │   ├── integration.py
│   │   │   └── sequence_packing.py
│   │   ├── heatmap/                        # 热力图模块
│   │   │   ├── diffusion_heatmap_head.py
│   │   │   └── diffusion/
│   │   └── action/                         # 动作模块
│   │       ├── diffusion_action_head.py
│   │       ├── transformer_action_head.py
│   │       ├── stop_head.py
│   │       └── progress_head.py
│   └── utils/
│       ├── loss.py
│       └── visualization.py
├── requirements.txt
└── README.md
```

---

## License

MIT License

---

## 参考

- [Qwen3-VL](https://github.com/QwenLM/Qwen-VL) - 视觉语言骨干模型
- [InternNav](https://github.com/OpenRobotLab/InternNav) - TransformerActionHead 参考实现
- [Diffusion Policy](https://github.com/real-stanford/diffusion_policy) - 扩散策略参考

# HeatmapVLN

本目录实现了一个用于 **第一人称跨帧热力图（inter-frame heatmap）** 与 **动作预测** 的训练/评估/推理流水线。
当前仓库内真实可用的入口脚本为：

- `scripts/train_history_action.py`：训练（四阶段 curriculum，包含 history/future 热力图 + action + stop）
- `scripts/evaluate.py`：评估（支持 history/future/action，支持保存可视化）
- `scripts/inference.py`：推理（支持对视频或数据集 clip 运行并保存热力图/动作）


---

## 快速开始

### 1) 环境安装

建议在 `Heatmap/` 目录下安装依赖：

```bash
cd Heatmap

# 建议 Python 3.11+（与本项目依赖/代码路径更匹配）
conda create -n models python=3.11 -y
conda activate models

pip install -U pip
pip install -r requirements.txt
```

如果你需要安装带 CUDA 的 PyTorch，请按你机器 CUDA 版本选择对应 wheel（`requirements.txt` 内也有注释提示）。

---

## 数据集准备（VLNSlidingWindowDataset）

训练/评估使用 `src/data/vln_sliding_window_dataset.py` 的 `VLNSlidingWindowDataset`。
配置文件中通过 `data.root` 指定数据集根目录，默认是：

- `configs/training_config_full_model.yaml` 里的 `data.root: /root/autodl-tmp/dataset_with_actions`

### 目录结构要求（非常重要）

数据按 split 组织（训练脚本固定 train 为 `train`，验证 split 可通过 `data.val_split` 指定，例如 `val_unseen`）：

```text
<data.root>/
  train/
    <scene_id>/
      clip_000000/
        meta.json
        poses.json
        rgb/
          000000.png
          000001.png
          ...
        depth/                    (可选)
          000000.npy
          000001.npy
          ...
        intrinsics.json           (可选)
        actions.npy               (可选，但训练 action head 强烈建议提供)
        discrete_actions.npy      (可选，训练 stop head 时建议提供)
  val_unseen/                     (或 val/)
    <scene_id>/
      clip_000000/
        ...
```

### 必需/可选文件说明

- **必需**
  - **`meta.json`**：至少包含 `num_frames`；可包含 `instruction`
  - **`poses.json`**：长度为 `num_frames` 的 4×4 位姿矩阵列表
  - **`rgb/000000.png`**：按 6 位零填充命名的 RGB 帧
- **可选**
  - **`depth/xxxxxx.npy`**：用于遮挡检测；缺失时会跳过遮挡判断
  - **`intrinsics.json`**：若存在用于提供原始图像宽高；否则默认 `(512, 256)`（全景图常见尺寸）
  - **`actions.npy`**：连续动作（dx, dy）。缺失时该样本 `action_valid=0`，动作置 0
  - **`discrete_actions.npy`**：离散动作（STOP=0, FORWARD=1, LEFT=2, RIGHT=3）。缺失时 stop 标签默认为非 stop

---

## 推理（Inference）

推理脚本：`scripts/inference.py`  
支持两种输入：

- **视频文件**：`--video /path/to/video.mp4`
- **数据集 clip 目录**：`--clip <data.root>/<split>/<scene>/clip_xxxxxx`

如果你不传 `--use-history/--use-future/--use-actions`，脚本会默认全部输出。

### 对数据集 clip 推理（推荐）

```bash
cd /root/VLN/Project

python scripts/inference.py \
  --clip /root/autodl-tmp/dataset_with_actions/val_unseen/<scene_id>/clip_000000 \
  --config configs/training_config_full_model.yaml \
  --output-dir ./outputs_inference
```

### 对视频推理

```bash
cd /root/VLN/Project

python scripts/inference.py \
  --video /path/to/video.mp4 \
  --instruction "从起点出发，沿走廊前进并找到目标" \
  --config configs/training_config_full_model.yaml \
  --output-dir ./outputs_inference
```

### 推理输出

默认会在 `--output-dir` 下生成：

- `*_history_heatmaps.png`（若启用 history）
- `*_future_heatmaps.png`（若启用 future）
- `*_actions.npy`（若启用 action）

---

## 训练（Training）

训练脚本：`scripts/train_history_action.py`  
默认读取：`--config configs/training_config_full_model.yaml`

### 一键按配置跑完整四阶段

```bash
cd /root/VLN/Project

python scripts/train_history_action.py \
  --config configs/training_config_full_model.yaml
```

### 常用训练参数

- **从检查点恢复**

```bash
python scripts/train_history_action.py \
  --config configs/training_config_full_model.yaml \
  --resume /path/to/ckpt.pth
```

- **自动从最新检查点恢复**

```bash
python scripts/train_history_action.py \
  --config configs/training_config_full_model.yaml \
  --auto-resume
```

- **只跑某个阶段（按名称或索引）**

```bash
# 例：只跑 joint_128 阶段
python scripts/train_history_action.py \
  --config configs/training_config_full_model.yaml \
  --stage joint_128 \
  --stage-only
```

- **调试：只构建模型与数据，不实际训练**

```bash
python scripts/train_history_action.py \
  --config configs/training_config_full_model.yaml \
  --dry-run
```

### 训练输出

训练输出目录由配置控制：

- `log.out_dir`（默认：`/root/autodl-tmp/vln_history_action_outputs`）

其中会包含 `train.log`、checkpoint、可视化图片、以及（可选）TensorBoard 日志。

---

## 评估（Evaluate）

评估脚本：`scripts/evaluate.py`  
必须提供 `--checkpoint`。

如果你不传 `--use-history/--use-future/--use-action`，脚本会默认全部评估。

```bash
cd /root/VLN/Project

python scripts/evaluate.py \
  --config configs/training_config_full_model.yaml \
  --checkpoint /path/to/ckpt.pth \
  --split val \
  --save-vis \
  --num-vis 20
```

---

## 配置文件说明（configs/training_config_full_model.yaml）

当前 `configs/` 目录只有一个配置：`training_config_full_model.yaml`。

你通常需要优先检查并修改：

- `data.root`：数据集根目录
- `data.val_split`：验证 split（例如 `val_unseen`）
- `model.llm.model_path`：本地 Qwen2.5-VL 权重路径（默认写在 `Project/models/qwen_2.5_vl`）
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

下面只列 **跑通训练/评估/推理最关键** 的文件与目录（像 `utils/` 这类辅助模块默认不展开）：

```text
Project/
  configs/
    training_config_full_model.yaml          # 唯一配置：数据路径、训练阶段、损失、日志等

  scripts/                                   # 三个入口脚本（README 命令都以它们为准）
    train_history_action.py                   # 训练：四阶段 curriculum（history/future heatmap + action + stop）
    evaluate.py                               # 评估：history/future/action + 可视化
    inference.py                              # 推理：对 video 或 dataset clip 生成 heatmap/actions

  src/
    data/
      vln_sliding_window_dataset.py           # 数据集：VLNSlidingWindowDataset（读取 meta/poses/rgb/depth/actions）
      keyframe_selector.py                    # 关键帧选择（与 pipeline 的采样策略相关）
      spatial_analysis.py                     # 空间新颖性/覆盖分析

    models/
      pipeline.py                             # 核心模型：SpatialMLLMPipeline + SpatialMLLMIntegrationConfig

      heatmap/
        diffusion_heatmap_head.py             # DiffusionHeatmapHead（history/future 双头）
        diffusion/                            # 扩散网络细节/组件
        generator.py                          # 热力图生成相关工具
        visualizer.py                         # 热力图可视化

      action/
        diffusion_action_head.py              # DiffusionActionHead（连续动作 dx,dy）
        stop_head.py                          # StopPredictionHead（二分类 STOP）
        action_config.py                      # 动作 head 配置/超参
        diffusion/                            # 扩散网络细节/组件

      llm/
        integration.py                        # LLM 兼容层/集成逻辑
        memory_efficient.py                   # 显存友好模式（可选）

      qwen2_5_vl/                             # Qwen2.5-VL 本地实现/适配（HF-style）
        modeling_qwen2_5_vl.py
        processing_qwen2_5_vl.py

      dinov3/                                 # DINOv3 特征抽取与兼容层
        compatibility.py

      vggt/                                   # VGGT 3D 编码器（第三方代码集成）
        vggt/                                 # VGGT 包主体
```



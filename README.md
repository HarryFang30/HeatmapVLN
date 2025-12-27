# HeatmapVLN

本目录实现了一个用于 **第一人称跨帧热力图（inter-frame heatmap）** 与 **动作预测** 的训练/评估/推理流水线。
当前仓库内真实可用的入口脚本为：

- `scripts/train_history_action.py`：训练（四阶段 curriculum，包含 history/future 热力图 + action + stop）
- `scripts/evaluate.py`：评估（支持 history/future/action，支持保存可视化）
- `scripts/inference.py`：推理（支持对视频或数据集 clip 运行并保存热力图/动作）


---

## 快速开始

### 1) 环境安装

建议在 `HeatmapVLN/` 目录下安装依赖：

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

## 关键帧选取算法

推理时，模型使用 **Greedy Maximum Coverage** 算法从输入视频中智能选取关键帧：

### 算法流程

1. **3D 几何提取**：所有帧通过 VGGT 提取 3D 世界点云 + 置信度
2. **自适应体素化**：将点云转换为统一的体素表示
3. **贪心选择**：迭代选择覆盖最多新体素的帧，最大化空间覆盖

### 配置参数

在 `configs/training_config_full_model.yaml` 中：

```yaml
model:
  target_keyframes: 16      # 选取的关键帧数 (N_k)
  total_frames: 128         # 输入候选帧数 (N_m)
  sampling_method: greedy_coverage  # 采样策略
```

### 数据流

```
视频 [N_m 帧] → VGGT → 3D 点云 → 体素化 → 贪心选择 → [N_k 关键帧]
                ↓
        关键帧 → DINOv3 + VGGT → 特征融合 → LLM → 热力图/动作
```

**注**：训练时由于显存限制通常加载帧数较少，关键帧选取会自动跳过。

---

## 数据集加载逻辑（VLNSlidingWindowDataset）

### 核心思想：滑动窗口扩展

**一段 T 帧视频 → 生成 (T - min_history) 个训练样本**

每个样本包含：
- **历史帧**：从视频开始到当前帧之前的帧（采样 K 帧）
- **当前帧**：第 i 帧作为当前观测
- **热力图**：历史帧相机位置在当前帧中的投影（高斯热力图）
- **动作标签**：从当前帧到下一帧的动作 (dx, dy) 和 stop 信号

### 数据流

```
视频 [T 帧]
    ↓ 滑动窗口
样本 0: 历史[0:0]  + 当前[0]  + 热力图 + 动作[0→1]
样本 1: 历史[0:1]  + 当前[1]  + 热力图 + 动作[1→2]
样本 2: 历史[0:2]  + 当前[2]  + 热力图 + 动作[2→3]
...
样本 T-1: 历史[0:T-1] + 当前[T-1] + 热力图 + 动作[T-1→T] (STOP)
```

### 关键参数

```python
VLNSlidingWindowDataset(
    root="/path/to/dataset",
    split="train",
    min_history=5,          # 最小历史帧数（T >= 5 才生成样本）
    num_history_sample=8,   # 从历史中采样的帧数 K
    image_size=(224, 224),  # 图像尺寸
    hm_size=(64, 64),       # 热力图尺寸
    sample_stride=1,        # 采样步长（1=每帧，5=每5帧）
)
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

## 数据集准备

### 数据集结构

训练/评估使用 `VLNSlidingWindowDataset`，配置文件通过 `data.root` 指定数据集根目录：

- 默认：`configs/training_config_full_model.yaml` 中的 `data.root: /root/autodl-tmp/dataset_with_actions`
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
```

### 必需/可选文件说明

| 文件 | 类型 | 说明 |
|------|------|------|
| `meta.json` | 必需 | 至少包含 `num_frames`（帧数）；可包含 `instruction`（导航指令） |
| `poses.json` | 必需 | 长度为 T 的 4×4 位姿矩阵列表 |
| `rgb/000000.png` | 必需 | 按 6 位零填充命名的 RGB 帧 |
| `depth/000000.npy` | 可选 | 用于遮挡检测；缺失时跳过遮挡判断 |
| `actions.npy` | 可选 | 连续动作 [T, 2] (dx, dy)；缺失时 `action_valid=0` |
| `discrete_actions.npy` | 可选 | 离散动作 [T] (STOP/FORWARD/LEFT/RIGHT)；缺失时 stop 标签默认非 stop |
| `intrinsics.json` | 可选 | 相机内参；缺失时使用默认全景图尺寸 (512, 256) |

**建议**：训练 action/stop head 时务必提供对应的动作文件。

---

## 推理（Inference）

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
  --config configs/training_config_full_model.yaml \
  --output-dir ./outputs_inference
```

### 对视频推理

```bash
cd HeatmapVLN

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
cd HeatmapVLN

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

- `log.out_dir`（默认：`vln_history_action_outputs`）

其中会包含 `train.log`、checkpoint、可视化图片、以及（可选）TensorBoard 日志。

---

## 评估（Evaluate）

评估脚本：`scripts/evaluate.py`  
必须提供 `--checkpoint`。

如果你不传 `--use-history/--use-future/--use-action`，脚本会默认全部评估。

```bash
cd HeatmapVLN

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
- `model.llm.model_path`：本地 Qwen2.5-VL 权重路径（默认写在 `models/qwen_2.5_vl`）
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
    training_config_full_model.yaml          # 唯一配置：数据路径、训练阶段、损失、日志等

  scripts/                                   # 三个入口脚本（README 命令都以它们为准）
    train_history_action.py                   # 训练：四阶段 curriculum（history/future heatmap + action + stop）
    evaluate.py                               # 评估：history/future/action + 可视化
    inference.py                              # 推理：对 video 或 dataset clip 生成 heatmap/actions

  src/
    data/
      vln_sliding_window_dataset.py           # 数据集：VLNSlidingWindowDataset（读取 meta/poses/rgb/depth/actions）
      keyframe_selector.py                    # 关键帧选择器（空间感知采样）
      frame_sampler.py                        # 贪心最大覆盖算法（体素化 + 增量选择）

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



# HeatmapVLN

面向 Vision-Language Navigation（VLN）的训练与分析仓库，当前默认工作流聚焦：

- 基于 `Qwen2.5-VL / InternNav backbone` 的视觉语言特征提取
- 基于 `HeatmapVLN v2` 的历史位置热力图预测
- 基于 `InternNav System 1 / NextDiTActionHead` 的 32 步轨迹预测

当前仓库里同时保留了两条配置路径：

- 推荐默认：`configs/train_config_internnav.yaml`
- 兼容保留：`configs/train_config.yaml`（Qwen3.5 路线，非当前默认共享环境）

旧版 `README.md` 中关于 Transformer/DDPM 轨迹头、进度头、静态架构图、根目录许可证等内容，已经不再代表当前仓库的默认状态；本文档只描述代码里目前真实存在且可直接核对的流程。

## 当前默认方案

| 组件 | 当前默认 |
| --- | --- |
| 主配置 | `configs/train_config_internnav.yaml` |
| VLM 骨干 | `models/internnav_backbone` |
| backbone 类型 | `qwen2_5_vl` |
| Heatmap 头 | `HeatmapVLN v2` |
| 轨迹头 | `NextDiTActionHead` |
| 默认轨迹长度 | 32 steps |
| 默认训练数据根目录 | `/workspace/r2r_panoramic_data` |
| 默认输出目录 | `/root/autodl-tmp/vln_training_outputs` |
| TensorBoard 目录 | `/root/tf-logs` |

## 仓库结构

```text
HeatmapVLN/
├── configs/                     # 训练配置
├── scripts/                     # 训练、评估、推理、可视化、模型转换脚本
├── src/
│   ├── data/                    # 数据集与 collator
│   ├── models/                  # VLNPipeline、Heatmap、NextDiT 等
│   └── utils/                   # 日志、通知、可视化等工具
├── docs/                        # 补充文档
├── docker/                      # Docker 相关脚本与说明
├── data/fgr2r/                  # FGR2R 原始说明与许可证
└── models/                      # 本地模型目录
```

主要入口脚本：

| 脚本 | 用途 |
| --- | --- |
| `scripts/train.py` | 训练主入口 |
| `scripts/evaluate.py` | 通用评估 |
| `scripts/eval_heatmap.py` | 热力图专项评估 |
| `scripts/inference.py` | 单视频/单 clip 轨迹推理 |
| `scripts/visualize_heatmap.py` | 4 视角热力图对比可视化 |
| `scripts/visualize_trajectory_heatmaps.py` | 全景轨迹热力图时序可视化 |
| `scripts/convert_internnav_backbone.py` | 拆分 InternNav backbone / System 1 权重 |
| `scripts/monitor_gpu_idle.py` | GPU 空闲监控与飞书提醒 |

## 环境要求

- Python `3.11`
- PyTorch `2.7.0`
- CUDA `12.8`
- `transformers==4.51.0`
- 默认注意力实现：`sdpa`

安装示例：

```bash
conda create -n heatmapvln python=3.11 -y
conda activate heatmapvln

pip install torch==2.7.0 torchvision==0.22.0 torchaudio==2.7.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

说明：

- 当前共享环境基线是 `transformers==4.51.0` + `numpy==1.26.4`。
- 现有默认配置优先使用 `sdpa`，不要把旧文档中的 FlashAttention 视为默认前提。
- 当前训练/评估路径不支持 `sequence packing`，请保持 `model.llm.enable_packing=false`。

## 权重准备

### 1. InternNav 路线（推荐）

默认配置依赖以下文件：

- `models/internnav_backbone/`
- `models/internnav_system1.safetensors`
- `models/depth_anything_v2_metric_hypersim_vits.pth`

如果你只有原始 InternNav 模型目录，可执行：

```bash
python scripts/convert_internnav_backbone.py \
  --src /workspace/InternNav_Model \
  --backbone-dst models/internnav_backbone \
  --system1-dst models/internnav_system1.safetensors
```

注意：

- 当前仓库中的 `models/internnav_backbone/model.safetensors.index.json` 指向 `model-00001-of-00004.safetensors` 等分片；如果这些分片实际不存在，说明 backbone 还没有准备完整。
- `convert_internnav_backbone.py` 只会生成 backbone 和 System 1；`Depth Anything v2` 权重仍需你自行放到配置指定位置。

### 2. Qwen3.5 路线（兼容保留）

`configs/train_config.yaml` 仍保留 Qwen3.5 路线，额外需要：

- `models/qwen_3.5`
- `models/dualvln_system1_pretrained.safetensors`

该路线不是当前 README 推荐默认路径，但代码仍可用。

## 数据集格式

`src/data/vln_sliding_window_dataset.py` 当前支持两类组织方式：

### 1. 标准 split 目录

```text
<data_root>/
├── train/
│   └── <scene_id>/
│       └── clip_000000/
└── val_unseen/
    └── <scene_id>/
        └── clip_000000/
```

### 2. 无 split 目录

```text
<data_root>/
└── <scene_id>/
    └── clip_000000/
```

如果没有 `train/val_*` 层，dataset 会自动按 scene 做哈希切分。

### 单个 clip 的常见内容

帧文件模式：

```text
clip_xxxxxx/
├── meta.json
├── poses.json
├── intrinsics.json             # 可选
├── rgb/
│   ├── front/                  # 全景数据时使用
│   ├── right/
│   ├── back/
│   └── left/
├── depth/                      # 可选
├── actions.npy
└── discrete_actions.npy
```

块文件模式：

```text
clip_xxxxxx/
├── meta.json
├── intrinsics.json             # 可选
├── chunks/
│   └── chunk_*.npz
├── actions.npy
└── discrete_actions.npy
```

补充说明：

- `meta.json.storage_format` 可显式标记 `frames` 或 `chunks`；如果未写，dataset 会按是否存在 `chunks/` 自动判断。
- 全景热力图链路要求 4 视角数据，dataset 会检测 `front/right/back/left` 或 chunk 内的 `rgb_front/rgb_right/rgb_back/rgb_left`。
- `actions.npy[i]` 表示从 `frame[i]` 到 `frame[i+1]` 的 agent-local 位移。
- `discrete_actions.npy` 约定为 `0=STOP, 1=MOVE_FORWARD, 2=TURN_LEFT, 3=TURN_RIGHT`。

### FGR2R 子指令

如果配置中开启：

```yaml
data:
  trajectory:
    use_subinstruction: true
```

则还需要准备：

- `data/fgr2r/subinstr_mapping.json.gz`

当前仓库只包含 `data/fgr2r/README.md` 和对应许可证说明，不包含这个映射文件本体。

## 快速验证

建议先做一次 dry-run：

```bash
python scripts/train.py --config configs/train_config_internnav.yaml --dry-run
```

再做一个极小规模训练冒烟：

```bash
python scripts/train.py \
  --config configs/train_config_internnav.yaml \
  --epochs 1 \
  --max-batches 2
```

## 训练

### 默认 InternNav 训练

```bash
python scripts/train.py --config configs/train_config_internnav.yaml
```

### 热力图专用训练

```bash
python scripts/train.py --config configs/train_heatmap_config.yaml
```

### Qwen3.5 兼容训练

```bash
python scripts/train.py --config configs/train_config.yaml
```

### 自动续训

```bash
python scripts/train.py --config configs/train_config_internnav.yaml --auto-resume
```

### 只加载权重，不恢复优化器状态

```bash
python scripts/train.py \
  --config configs/train_config_internnav.yaml \
  --load-weights /path/to/checkpoint.pth
```

### 多卡训练

```bash
torchrun --nproc_per_node=2 scripts/train.py \
  --config configs/train_config_internnav.yaml \
  --distributed
```

使用多卡时请同时确认：

- `gpu.multi_gpu.enabled=true`
- `WORLD_SIZE` 与 `torchrun` 启动参数一致

### 当前默认训练策略

`configs/train_config_internnav.yaml` 采用的是“桥接层适配”思路：

- 训练 `heatmap_vln`
- 训练 `latent_queries`
- 训练 `cond_projector`
- 训练 `llm_projector`
- 训练 `lora`
- 冻结 NextDiT System 1 核心模块

这和旧 README 中“Transformer Decoder + DDPM 全量主路径”已经不是一回事。

## 评估

### 通用评估

```bash
python scripts/evaluate.py \
  --config configs/train_config_internnav.yaml \
  --checkpoint /path/to/best.pth \
  --split val_unseen \
  --save-vis
```

### 热力图专项评估

```bash
python scripts/eval_heatmap.py \
  --config configs/train_heatmap_config.yaml \
  --checkpoint /path/to/best.pth \
  --max-samples 200
```

说明：

- `scripts/evaluate.py` 当前主要评估 heatmap 和 trajectory。
- 命令行里仍保留 `--eval-progress`，但当前脚本主体并不会实际产出 progress 指标，不应再把它视为成熟默认能力。

## 推理与可视化

### 1. 单视频 / 单 clip 轨迹推理

```bash
python scripts/inference.py \
  --config configs/train_config_internnav.yaml \
  --checkpoint /path/to/best.pth \
  --video /path/to/video.mp4 \
  --instruction "Go forward and turn right at the door" \
  --output-dir ./outputs_inference
```

或：

```bash
python scripts/inference.py \
  --config configs/train_config_internnav.yaml \
  --checkpoint /path/to/best.pth \
  --clip /path/to/clip_dir \
  --output-dir ./outputs_inference
```

重要限制：

- 当前 `scripts/inference.py` 只接受单路视频/clip 帧。
- 即使传入 `--output-heatmap`，脚本也无法为 HeatmapVLN v2 构造全景 `current_views/history_panoramas`，因此不能作为热力图推理入口。

### 2. 4 视角热力图可视化

```bash
python scripts/visualize_heatmap.py \
  --checkpoint /path/to/best.pth \
  --num-samples 10 \
  --output-dir ./vis_heatmap_4view
```

这个脚本会复用训练时的数据加载逻辑，更适合检查：

- GT 热力图
- 预测热力图
- 可见性输出
- 4 视角 overlay 效果

### 3. 轨迹热力图时序可视化

```bash
python scripts/visualize_trajectory_heatmaps.py \
  --checkpoint /path/to/best.pth \
  --num-clips 3 \
  --frames-per-clip 32 \
  --output-dir ./vis_trajectory
```

补充说明：

- 该脚本要求数据集包含全景 4 视角。
- 如果 clip 目录中存在 `topdown_trajectory.jpg` 和 `topdown_transform.json`，会额外渲染 BEV 俯视图。

## 训练产物

每次训练都会在 `log.out_dir` 下创建独立 run 目录，并维护 `latest` 软链接：

```text
run_YYYYMMDD_HHMMSS/
├── manifest/
├── logs/
├── checkpoints/
├── visualizations/
├── plots/
└── tensorboard/
```

常用查看方式：

```bash
tensorboard --logdir /root/tf-logs --port=6006
```

详细说明见：

- `docs/training_outputs.md`
- `docs/loss.md`
- `docs/heatmap_loss_strategy.md`
- `docs/troubleshooting-guide.md`
- `docs/HeatmapVLN完整设计.md`

## Docker

仓库提供了 `docker/` 目录和启动菜单脚本：

```bash
./docker/docker-run.sh
```

但要注意：

- `docker/` 目录中的部分脚本和说明仍沿用 `dataset_with_actions`、`vln_training_outputs` 等旧宿主目录命名。
- 实际使用时，必须保证容器内路径和配置文件中的 `data.root`、`log.out_dir`、`log.tensorboard_dir` 对齐。
- 如果你以 `configs/train_config_internnav.yaml` 为主，请优先围绕 `/workspace/r2r_panoramic_data` 来组织容器挂载。

## 常见注意事项

### 1. 飞书通知配置

多个 YAML 配置里都启用了 `log.notify.enabled: true` 并带有 webhook 示例值。正式使用前请：

- 替换为你自己的 webhook
- 或直接关闭通知

### 2. `monitor_gpu_idle.py` 的默认占卡脚本路径

该脚本默认使用：

- `/workspace/train.py`

而当前仓库训练入口实际在：

- `scripts/train.py`

如果你要在本仓库内直接使用它，建议显式指定：

```bash
python scripts/monitor_gpu_idle.py \
  --occupy-script /workspace/HeatmapVLN/scripts/train.py
```

### 3. 旧 README 里提到的资源并不都还存在

当前仓库中：

- `assets/architecture.png` 不存在
- 根目录 `LICENSE` 不存在

因此本文档不再引用这些资源。

## 相关补充

- `data/fgr2r/README.md`：FGR2R 原始数据说明
- `data/fgr2r/LICENSE`：FGR2R 数据许可证

如果后续你准备继续整理文档，优先建议同步检查：

- `docker/DOCKER.md`
- `docker/README.md`

这两份文档仍保留了一些旧目录命名和旧命令示例。

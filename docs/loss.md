# HeatmapVLN Loss 设计

## 任务定义

给定当前全景观测（前/右/后/左四个方向），预测每个历史帧拍摄位置在当前视图中的投影热力图。

模型输出：
- `visibility`: `(N_hist, 4)` — 每个历史帧在 4 个方向上是否可见（logits）
- `heatmaps`: `(N_hist, 4, 64, 64)` — 每个方向上的定位热力图（sigmoid 激活，值域 0~1）

核心难点：模型必须区分"看过的场景"和"历史帧拍摄的位置"，不应把视觉相似的区域也激活。

## 设计原则

- Backbone 同时接收两种互补信号：peak CE（可见视图定位）+ neg BCE（不可见视图压零）
- vis_head 独立判断可见性（加权 BCE），梯度与 backbone 隔离（hm_flat.detach()）
- 推理时 vis_gate 门控热力图输出

## 训练与推理的语义对齐

训练时用 Softmax CE，所以推理时也用 spatial softmax 概率（而非 sigmoid 值）。

```
训练: logit(sigmoid_output) → softmax → CE loss
推理: logit(sigmoid_output) → softmax → 概率图 × sigmoid(vis) 门控
```

这消除了 train-inference 语义鸿沟。Softmax CE 是尺度不变的（所有 logit 同时偏移不改变分布），
所以 sigmoid 值可能很小（如 0.01），但 softmax 概率分布始终有效。诊断指标看 softmax
peak ratio（相对均匀分布的倍数），而非 sigmoid max。

## Loss 组成

### 1. Visibility BCE（加权）

```
vis_loss = BCEWithLogits(pred_vis, gt_vis, pos_weight=7.0)
```

训练 visibility head，推理时用 `sigmoid(vis)` 门控热力图。
`pos_weight=7.0` 修正 87%/13% 的类别不平衡（neg/pos ≈ 7:1），
确保 vis_head 不会坍缩到全部预测"不可见"。

vis_head 输入：query(c_llm) + coarse_heatmap_flat(H*W).detach()，
梯度不回传到 backbone。

### 2. Softmax Cross-Entropy — "目标在哪个像素？"

```
logits = logit(pred_sigmoid)
gt_prob = gt / gt.sum()
ce_loss = F.cross_entropy(logits.reshape(K,-1), gt_prob.reshape(K,-1))
```

- 只作用于 gt\_vis > 0 的可见视图（~13%）
- 4096 像素类分类，像素竞争防止正样本内假阳性

### 3. Coordinate Loss — 辅助定位

```
coord_loss = euclidean_distance(soft_argmax(pred), soft_argmax(gt))
```

低权重辅助项。

### 4. Neg Loss — 不可见视图压零

```
neg_loss = BCEWithLogits(logit(pred_neg), zeros)
```

- 只作用于 gt_vis = 0 的不可见视图（~87%），对每个像素施加 BCE 推向 0
- 提供密集梯度信号（每个像素都有梯度），帮助 backbone 学到正确表征
- 与 peak CE 作用在不同视图上，没有梯度冲突——两者互补：
  - peak CE 教 backbone "有目标时在哪激活"
  - neg loss 教 backbone "没目标时全部压零"

## 权重配置

```yaml
lambda_vis:        1.0   # visibility BCE
lambda_peak:       1.0   # softmax CE
lambda_coord:      0.2   # coordinate loss
lambda_neg:        1.0   # per-pixel BCE on invisible views
vis_pos_weight:    7.0   # visibility BCE pos_weight
```

## 推理输出

```python
# heatmap_vln.py: _gated_softmax_heatmaps()
logits = logit(sigmoid_heatmaps)
probs = softmax(logits over H×W)      # 每个视图的概率分布
output = probs × sigmoid(visibility)   # 门控后的概率图
```

可视化时自动归一化到 [0,1] 范围（softmax 概率值小但分布有效）。

## Loss 演进历史

| 版本 | 正样本定位 | 负样本抑制 | 推理空间 | 问题 |
|------|-----------|-----------|---------|------|
| v1 | KL + coord(λ=5) | L2(λ=0.1) | sigmoid | 权重失衡，假阳性爆炸 |
| v4 | QFL(β=2) | QFL 统一 | sigmoid | 假阳性梯度不够强 |
| v5 | Softmax CE | per-pixel BCE | sigmoid | sigmoid 坍缩（CE 尺度不变） |
| v6 | CE + L1 magnitude | per-pixel BCE | sigmoid | 不够干净，用 L1 补 CE 盲区 |
| v7 | Softmax CE | per-pixel BCE | softmax | 训推语义对齐，vis_head 坍缩 |
| v8 | Softmax CE | Uniform CE (KL→uniform) | softmax | 梯度竞争导致坍缩 |
| v9 | Softmax CE | 无（vis_gate 负责） | softmax | 从头训练坍缩（梯度密度不足） |
| **v10** | **Softmax CE** | **per-pixel BCE + vis_gate** | **softmax** | **当前版本** |

### v7 → v10 核心差异

v10 本质上回归了 v7 的 loss 设计（peak CE + neg BCE），但修复了 vis_head：

| 改进 | v7 | v10 |
|------|-----|------|
| vis_head 架构 | Linear(c_llm+3, 128) | Linear(c_llm+64, 256) |
| vis_head 输入 | 3 个标量(max/mean/std) | 完整 8x8 coarse heatmap |
| vis_head BCE | 无 pos_weight（坍缩） | pos_weight=7.0 |
| vis_head 学习率 | 共享 backbone LR | 独立 vis_head_lr=1e-3 |
| hm_flat 梯度 | 回传到 backbone | **.detach()** 隔离 |

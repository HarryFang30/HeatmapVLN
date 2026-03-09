# HeatmapVLN Loss 设计

## 任务定义

给定当前全景观测（前/右/后/左四个方向），预测每个历史帧拍摄位置在当前视图中的投影热力图。

模型输出：
- `visibility`: `(N_hist, 4)` — 每个历史帧在 4 个方向上是否可见（logits）
- `heatmaps`: `(N_hist, 4, 64, 64)` — 每个方向上的定位热力图（sigmoid 激活，值域 0~1）

核心难点：模型必须区分"看过的场景"和"历史帧拍摄的位置"，不应把视觉相似的区域也激活。

## 训练与推理的语义对齐

训练时用 Softmax CE，所以推理时也用 spatial softmax 概率（而非 sigmoid 值）。

```
训练: logit(sigmoid_output) → softmax → CE loss
推理: logit(sigmoid_output) → softmax → 概率图 × sigmoid(vis) 门控
```

这消除了 train-inference 语义鸿沟。Softmax CE 是尺度不变的（所有 logit 同时偏移不改变分布），
所以 sigmoid 值可能很小（如 0.01），但 softmax 概率分布始终有效。诊断指标看 softmax
peak ratio（相对均匀分布的倍数），而非 sigmoid max。

## 四项 Loss

### 1. Visibility BCE

```
vis_loss = BCEWithLogits(pred_vis, gt_vis)
```

训练 visibility head，推理时用 `sigmoid(vis)` 门控热力图。

### 2. Softmax Cross-Entropy — "目标在哪个像素？"

```
logits = logit(pred_sigmoid)
gt_prob = gt / gt.sum()
ce_loss = F.cross_entropy(logits.reshape(K,-1), gt_prob.reshape(K,-1))
```

- 只作用于 gt\_vis > 0 的视图
- 4096 像素类分类，像素竞争防止正样本内假阳性
- 分布保证始终有效，不会坍缩

### 3. Negative BCE — "不可见方向必须全黑"

```
neg_loss = -log1p(-pred)  # 等价 BCE(pred, 0)
```

- 只作用于 gt\_vis = 0 的视图
- 对 visibility gate 的纵深防御

### 4. Coordinate Loss — 辅助定位

```
coord_loss = euclidean_distance(soft_argmax(pred), soft_argmax(gt))
```

低权重辅助项。

## 权重配置

```yaml
lambda_vis:   1.0   # visibility BCE
lambda_peak:  1.0   # softmax CE
lambda_neg:   1.0   # negative BCE
lambda_coord: 0.2   # coordinate loss
lambda_kl:    0.0   # 未使用
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
| **v7** | **Softmax CE** | **per-pixel BCE** | **softmax** | **训推语义对齐** |

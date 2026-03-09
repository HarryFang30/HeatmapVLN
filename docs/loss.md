# HeatmapVLN Loss 设计

## 任务定义

给定当前全景观测（前/右/后/左四个方向），预测每个历史帧拍摄位置在当前视图中的投影热力图。

模型输出：
- `visibility`: `(N_hist, 4)` — 每个历史帧在 4 个方向上是否可见（logits）
- `heatmaps`: `(N_hist, 4, 64, 64)` — 每个方向上的定位热力图（sigmoid 激活，值域 0~1）

核心难点：模型必须区分"看过的场景"和"历史帧拍摄的位置"，不应把视觉相似的区域也激活。

## 设计思路

三个不同的子问题，用不同的 loss：

| 子问题 | 适用视图 | Loss 类型 |
|--------|----------|-----------|
| "目标在哪里？" | gt\_vis > 0（可见） | Softmax Cross-Entropy |
| "应该多亮？" | gt\_vis > 0（可见） | Pixel-level L1 |
| "这里不该亮" | gt\_vis = 0（不可见） | Per-pixel BCE → 0 |

## 五项 Loss

### 1. Visibility BCE — "这个方向看得到吗？"

```
vis_loss = BCEWithLogits(pred_vis, gt_vis)
```

- 作用于所有真实（非 padding）视图
- 训练 visibility head 区分可见/不可见方向
- 推理时用 `sigmoid(vis)` 作为门控乘到热力图上

### 2. Softmax Cross-Entropy — "可见的方向里，目标在哪个像素？"

```
gt_prob = gt / gt.sum()      # GT 归一化为概率分布
logits = logit(pred_sigmoid)  # 还原 raw logits
ce_loss = F.cross_entropy(logits, gt_prob)  # fused kernel
```

- 只作用于 gt\_vis > 0 的视图
- 将 64×64 = 4096 个像素视为 4096 类分类问题
- 像素竞争机制防止正样本内假阳性
- **尺度不变**：softmax 只看相对差异，不约束 sigmoid 绝对值

### 3. Magnitude L1 — "可见方向的输出应该多亮？"

```
mag_loss = L1(pred_pos, gt_pos)
```

- 只作用于 gt\_vis > 0 的视图
- **防坍缩的关键**：Softmax CE 是尺度不变的——所有 logit 同时下移不改变 CE。
  neg\_loss 持续在 ~75% 视图推输出向 0。没有 magnitude loss，没有任何力量推正样本向上，
  导致所有 sigmoid 输出坍缩到 ~0.01。
- L1 直接在 sigmoid 空间监督绝对值，提供唯一的上推梯度
- 坍缩时 L1 ≈ 0.015（有梯度），正确时 L1 ≈ 0（不扰动）

### 4. Negative BCE — "不可见的方向，热力图必须全黑"

```
neg_loss = -log1p(-pred)  # 等价于 BCE(pred, 0)，但更高效
```

- 只作用于 gt\_vis = 0 的视图
- 用 `torch.log1p` 直接计算，避免 logit→zeros\_like→bce\_with\_logits 的冗余链
- 梯度 ≈ pred：pred=0.9 时梯度强（0.9），pred=0.01 时几乎不扰动（0.01）

### 5. Coordinate Loss — 辅助定位微调

```
coord_loss = euclidean_distance(soft_argmax(pred), soft_argmax(gt))
```

- 低权重辅助项，补充 CE 的坐标级监督

## 权重配置

```yaml
lambda_vis:   1.0   # visibility BCE
lambda_peak:  1.0   # softmax CE（分布定位）
lambda_kl:    2.0   # magnitude L1（防坍缩）
lambda_neg:   1.0   # negative BCE（负样本压零）
lambda_coord: 0.2   # soft-argmax 坐标距离
```

```
total = vis + ce + 2.0×mag + neg + 0.2×coord
```

## 工程细节

### Train/Eval 一致性

模型始终在 `heatmaps` 键返回 raw h\_loc（未门控）。eval 模式下额外返回 `heatmaps_gated = h_loc × sigmoid(vis)`。Loss 始终作用在 raw h\_loc 上。

### Padding 屏蔽

通过 `history_mask` 排除 padding 历史帧的 loss 污染。

### 推理时的门控

```
h_final = sigmoid(visibility) × h_loc
```

## Loss 演进历史

| 版本 | 正样本定位 | 负样本抑制 | 问题 |
|------|-----------|-----------|------|
| v1 | KL + L2 + coord(λ=5) | L2(λ=0.1) + temp annealing | 权重失衡 50:1，假阳性爆炸 |
| v2 | coord + BCE peak | BCE + max penalty | AMP 不兼容，热力图坍缩 |
| v3 | coord + L1 peak(top-k) | pixel-level L2 | 梯度失衡（15000↓ vs 40↑），不收敛 |
| v4 | QFL(β=2) | QFL 统一处理 | 假阳性 pred≈0.9 时 L2 梯度仍不够强 |
| v5 | Softmax CE | per-pixel BCE | sigmoid 坍缩（CE 尺度不变，无上推力） |
| **v6（当前）** | **Softmax CE + L1** | **per-pixel BCE** | CE 管分布 + L1 管幅值 |

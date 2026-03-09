# HeatmapVLN Loss 设计

## 任务定义

给定当前全景观测（前/右/后/左四个方向），预测每个历史帧拍摄位置在当前视图中的投影热力图。

模型输出：
- `visibility`: `(N_hist, 4)` — 每个历史帧在 4 个方向上是否可见（logits）
- `heatmaps`: `(N_hist, 4, 64, 64)` — 每个方向上的定位热力图（sigmoid 激活，值域 0\~1）

核心难点：模型必须区分"看过的场景"和"历史帧拍摄的位置"，不应把视觉相似的区域也激活。

## 设计思路

两个不同的子问题，用不同的 loss：

| 子问题 | 适用视图 | Loss 类型 |
|--------|----------|-----------|
| "目标在哪里？" | gt\_vis > 0（可见） | Softmax Cross-Entropy |
| "这里不该亮" | gt\_vis = 0（不可见） | Per-pixel BCE → 0 |

## 四项 Loss

### 1. Visibility BCE — "这个方向看得到吗？"

```
vis_loss = BCEWithLogits(pred_vis, gt_vis)
```

- 作用于所有真实（非 padding）视图
- 训练 visibility head 区分可见/不可见方向
- 推理时用 `sigmoid(vis)` 作为门控乘到热力图上

### 2. Softmax Cross-Entropy — "可见的方向里，目标在哪个像素？"

```
gt_prob = gt / gt.sum()          # GT 归一化为概率分布（和为1）
logits = logit(pred_sigmoid)     # sigmoid 输出还原为 raw logits
log_probs = log_softmax(logits)  # 4096 个像素类的对数概率
ce_loss = -Σ gt_prob(x) · log_probs(x)
```

- 只作用于 gt\_vis > 0 的视图
- 将 64×64 = 4096 个像素视为 4096 类分类问题
- GT 高斯热力图归一化为概率分布（所有像素和为 1）

**为什么比 QFL/L2 更适合定位：**

- **像素竞争**：softmax 保证概率和为 1，提高一个像素必然压低其他，结构性防止正样本内的假阳性
- **无坍缩风险**：概率质量必须放在某个位置，不会收敛到全零
- **无条带伪影**：把质量铺成条带的代价等同于正确高斯，没有偷懒激励
- **梯度干净**：∂L/∂z\_i = softmax(z\_i) - gt\_prob(i)，每个像素自动获得正确方向和大小的梯度

### 3. Negative BCE — "不可见的方向，热力图必须全黑"

```
neg_logits = logit(pred_sigmoid)
neg_loss = BCEWithLogits(neg_logits, zeros)
```

- 只作用于 gt\_vis = 0 的视图
- Softmax CE 无法覆盖不可见视图（GT 全零，无法归一化为概率分布）
- 用 per-pixel BCE 直接推所有像素趋近 0
- 梯度 = sigmoid(z) = pred：pred=0.9 时梯度 0.9（强压），pred=0.01 时梯度 0.01（几乎不扰动）
- 对 visibility gate 的纵深防御——即使 vis head 判断失误，h\_loc 自身也被训练为全暗

### 4. Coordinate Loss — 辅助定位微调

```
pred_xy = soft_argmax(pred, temperature=1.0)
gt_xy   = soft_argmax(gt,   temperature=1.0)
coord_loss = euclidean_distance(pred_xy, gt_xy)
```

- 只作用于 gt\_vis > 0 的视图
- 低权重辅助项，提供显式的坐标级监督
- 补充 softmax CE：CE 监督分布形状，coord 监督峰值位置

## 权重配置

```yaml
lambda_vis:   1.0   # visibility BCE
lambda_peak:  1.0   # softmax CE（正样本定位）
lambda_neg:   1.0   # negative BCE（负样本压零）
lambda_coord: 0.2   # soft-argmax 坐标距离（辅助）
lambda_kl:    0.0   # 未使用，保留兼容
```

```
total = 1.0 × vis_loss + 1.0 × ce_loss + 1.0 × neg_loss + 0.2 × coord_loss
```

## 工程细节

### Train/Eval 一致性

模型始终在 `heatmaps` 键返回 raw h\_loc（未门控）。eval 模式下额外返回 `heatmaps_gated = h_loc × sigmoid(vis)`。Loss 始终作用在 raw h\_loc 上，确保训练和验证的损失语义一致。

### Padding 屏蔽

通过 `history_mask` 排除 padding 历史帧对所有 loss 的污染：
- vis\_loss 中排除 padding 位（否则 `BCEWithLogits(0, 0) = 0.693` 常数噪声）
- pos\_mask / neg\_mask 中排除 padding 位
- 当 `load_history_frames=false` 导致 mask 形状与模型输出不匹配时，安全跳过 masking

### 推理时的门控

推理时热力图最终输出为：

```
h_final = sigmoid(visibility) × h_loc
```

不可见方向由 visibility gate 压制，可见方向保留 raw h\_loc 的定位结果。

## Loss 演进历史

| 版本 | 正样本定位 | 负样本抑制 | 问题 |
|------|-----------|-----------|------|
| v1 | KL + L2 + coord(λ=5) | L2(λ=0.1) + temp annealing | 权重失衡 50:1，假阳性爆炸 |
| v2 | coord + BCE peak | BCE + max penalty | AMP 不兼容，热力图坍缩 |
| v3 | coord + L1 peak(top-k) | pixel-level L2 | 梯度失衡（15000↓ vs 40↑），不收敛 |
| v4 | QFL(β=2) | QFL 统一处理 | 假阳性 pred≈0.9 时 L2 梯度仍不够强 |
| **v5（当前）** | **Softmax CE** | **per-pixel BCE** | — |

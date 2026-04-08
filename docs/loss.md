# HeatmapVLN 当前 Loss 设计

本文档描述仓库里**当前主训练路径真实生效**的 loss 设计，以代码实现为准：

- `src/models/heatmap/heatmap_vln_loss.py`
- `scripts/training/train_loop.py`
- `scripts/training/validate.py`
- `src/models/action/nextdit_action_head.py`

不再把早期实验方案、设计提案或未接入训练循环的 loss 当作“当前版本”。

## 任务定义

给定当前全景观测（前/右/后/左四个方向），预测每个历史帧拍摄位置在当前视图中的投影热力图。

热力图分支当前输出：

- `visibility`: `(B, N_hist, 4)` 或 `(N_hist, 4)`，表示每个历史帧在 4 个方向上的可见性 logits
- `heatmaps`: `(B, N_hist, 4, 64, 64)` 或 `(N_hist, 4, 64, 64)`，表示每个方向上的定位热力图，值域为 `(0, 1)`，来自 fine head 最后的 `sigmoid`

在 `eval` / `inference` 阶段，模型还会额外输出：

- `heatmaps_gated`: 对 `heatmaps` 先做空间 softmax，再乘 `sigmoid(visibility)` 的结果

## 当前真实训练目标

主训练循环会先计算 `HeatmapVLNLoss`，再按配置可选叠加轨迹损失：

```text
total_loss
= heatmap_weight * heatmap_loss
+ trajectory_weight * trajectory_loss
```

其中：

- `heatmap_loss` 来自 `HeatmapVLNLoss`
- `trajectory_loss` 来自 `NextDiTActionHead.compute_loss`

默认热力图专用配置 `configs/train_heatmap_config.yaml` 中：

- `heatmap_weight = 1.0`
- `trajectory_weight = 0.0`

默认联合训练配置 `configs/train_config*.yaml` 中通常为：

- `heatmap_weight = 1.0`
- `trajectory_weight = 1.0`

## HeatmapVLNLoss 组成

`HeatmapVLNLoss` 由 4 项组成：

```text
L_hm
= lambda_vis   * L_vis
+ lambda_peak  * L_peak
+ lambda_coord * L_coord
+ lambda_neg   * L_neg
```

### 1. `L_vis`: 可见性加权 BCE

公式：

```text
L_vis = BCEWithLogits(pred_vis, gt_vis, pos_weight=vis_pos_weight)
```

实现细节：

- 使用 `F.binary_cross_entropy_with_logits(..., reduction="none")`
- 如果 batch 中存在 `history_mask`，会先屏蔽 padding 历史帧，再做有效位置平均
- `gt_vis` 优先使用数据集提供的 `gt_visibility`
- 若 batch 不含 `gt_visibility`，则退化为 `gt_heatmap.amax(dim=(-2,-1)).clamp(0,1)`

当前默认：

- `vis_pos_weight = 7.0`

说明：

- 这是为了解决正负样本不平衡，避免 visibility head 退化为“全部不可见”
- 推理时 `sigmoid(visibility)` 会作为热力图门控因子

### 2. `L_peak`: 可见视图上的 spatial softmax cross-entropy

公式：

```text
logits  = logit(pred_sigmoid)
gt_prob = gt / gt.sum()
L_peak  = CrossEntropy(logits, gt_prob)
```

实现细节：

- 只在 `gt_vis == 1` 的可见视图上计算
- 每张 `64x64` 热力图会被视为一个 `4096` 类分类问题
- 预测端不是直接对 `sigmoid` 值做 CE，而是先 `logit` 回原 logits，再做 spatial softmax
- GT 热力图会先按像素和归一化为概率分布

这意味着：

- `L_peak` 的理论最优值通常**不是 0**
- 当预测分布与 GT 分布完全一致时，`L_peak` 等于 GT 本身的熵
- 随机均匀预测时，`L_peak ≈ log(64 * 64) = log(4096) ≈ 8.318`

### 3. `L_coord`: soft-argmax 坐标辅助损失

公式：

```text
pred_weights   = softmax(pred * temperature)
target_weights = softmax(gt   * temperature)
L_coord        = mean(||coord(pred_weights) - coord(target_weights)||_2)
```

实现细节：

- 只在 `gt_vis == 1` 的可见视图上计算
- `temperature` 只影响这一项，不影响 `L_peak`
- 当前实现对 GT 使用的是 `softmax(raw_gt)`，不是先对 GT 高斯归一化再直接取质心
- 代码里带有一个 `1e-6` 的稳定项，因此理论最小值接近 `0.001`，而不是严格的 `0`

说明：

- 这是一个低权重辅助项，用来提供峰值位置牵引
- 它不是主监督项，主监督仍然是 `L_peak`

### 4. `L_neg`: 不可见视图压零项

公式：

```text
L_neg = mean(-log(1 - pred_neg))
```

其中 `pred_neg` 为不可见视图上的预测热力图值。

实现细节：

- 只在 `gt_vis == 0` 的不可见视图上计算
- 如果有 `history_mask`，同样会先排除 padding 历史帧
- 实现上等价于对目标 0 的逐像素 BCE，只是直接写成了 `-log1p(-pred)`

说明：

- 它为负样本提供了稠密梯度
- 与 `L_peak` 分别作用在不可见视图和可见视图上，语义上互补
- 当前训练配置常见做法是训练时开启，验证时关闭

## 训练与验证的一个重要差异

训练时：

- `lambda_neg` 从配置读取

验证时：

- `validate.py` 会强制使用 `lambda_neg = 0.0`

也就是说：

- `train/heatmap_loss` 和 `val/heatmap_loss` 不是完全同构指标
- 如果训练启用了 `L_neg`，训练 loss 天然会比验证多出一项

## 推理语义

当前训练和推理在主定位语义上是对齐的：

```text
训练: logit(sigmoid_heatmap) -> spatial softmax -> soft-target CE
推理: logit(sigmoid_heatmap) -> spatial softmax -> prob map
推理: prob map * sigmoid(visibility) -> gated output
```

因此：

- 训练时真正监督的是空间概率分布
- 推理时真正使用的也是空间概率分布，而不是原始 sigmoid 图

## 轨迹分支 loss

如果 `train_action = true` 且模型中启用了 `nextdit_action_head`，总 loss 还会叠加轨迹分支：

```text
L_traj = MSE(noise_pred, noise - gt_trajectory)
```

实现位于 `src/models/action/nextdit_action_head.py`，本质是 Flow Matching velocity prediction MSE。

实现细节：

- 先采样高斯噪声 `noise`
- 采样随机时间步 `u`
- 构造插值轨迹 `X_u = (1 - sigma) * X_0 + sigma * noise`
- 预测速度项，监督目标为 `noise - gt_trajectory`
- 若 batch 提供 `trajectory_valid`，会按样本 mask 后再取平均

理论上：

- 最优值是 `0`
- 随机零预测的基线不会是 `0`，通常至少在 `1` 以上，并取决于真实轨迹方差

## 与旧文档最容易混淆的几点

### 1. `src/utils/loss.py` 不是当前主训练路径

仓库里虽然还保留了很多历史 / 试验版 loss：

- `NavigationHeatmapLoss`
- `HighFreqHeatmapLoss`
- `FocalHeatmapLoss`
- `CombinedNavigationLoss`

但当前 `train_loop.py` / `validate.py` 并不会根据 `heatmap_loss_type` 切换到这些实现，主路径始终实例化 `HeatmapVLNLoss`。

### 2. “vis_head 与 backbone 梯度隔离”只对旧 coarse 头完整成立

如果关闭 trajectory attention，回退到 `CoarseLocalization`，其中确实存在：

- `hm_flat = heatmaps.flatten(-2).detach()`

此时 vis head 不会通过 coarse heatmap 反传回 backbone。

但当前默认配置启用了：

- `TrajectoryGuidedAttention`

它的 `visibility` 和 `coarse_heatmap` 共用同一个 transformer 编码过程，不再是旧版那种显式 `detach` 分离结构。

因此文档里如果泛化地写“vis_head 梯度与 backbone 完全隔离”，就和当前默认实现不一致了。

## 当前推荐阅读顺序

若要理解当前 loss 的真实行为，建议按这个顺序读代码：

1. `src/models/heatmap/heatmap_vln_loss.py`
2. `scripts/training/train_loop.py`
3. `scripts/training/validate.py`
4. `src/models/heatmap/heatmap_vln.py`
5. `src/models/action/nextdit_action_head.py`

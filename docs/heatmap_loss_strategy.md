# HeatmapVLN 热力图 Loss 策略说明

## 当前状态

这份文档原本记录的是一版**面向未来的 loss 改进提案**，其中包含：

- `QFL`
- `L_pos_mass`
- 对最终 gated heatmap 直接监督
- 分阶段 / warmup 训练建议

这些内容**不是当前主训练路径正在使用的实现**。

当前仓库真实生效的主路径请以 `docs/loss.md` 为准。那里记录的是：

- `HeatmapVLNLoss`
- `visibility BCE`
- `spatial softmax CE`
- `soft-argmax coord loss`
- `neg loss`
- 可选 `NextDiT` 轨迹 MSE

## 为什么保留这份文档

保留本文件的原因是：

- 其中有一些对失败模式的分析仍然有参考价值
- 它记录了团队曾经考虑过的替代设计方向
- 便于以后重新评估是否要引入更强的正样本局部质量约束

但需要明确：

- 本文件是**历史策略讨论**
- 不是当前实现说明
- 不应用它来解释训练日志中的实际 loss 数值

## 当前建议

如果你想了解“现在代码里到底在训练什么”，请优先阅读：

1. `docs/loss.md`
2. `src/models/heatmap/heatmap_vln_loss.py`
3. `scripts/training/train_loop.py`
4. `scripts/training/validate.py`

如果后续重新尝试：

- gated map 直接监督
- `L_pos_mass`
- 更强的正样本质量约束

再以本文件为起点重新整理会更合适。

## 历史提案的定位

这份提案主要关注的是以下问题：

- 全黑坍缩
- 正样本峰值不足
- 单像素尖峰
- 条带状投机解

这些担忧在理论上仍然成立，但它们描述的是“可能的改进方向”，不是当前 `HeatmapVLNLoss` 的正式定义。

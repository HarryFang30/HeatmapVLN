# 热力图分布优化方案

## 问题分析

### 现象
- 热力图点 **89.4%** 集中在垂直中间行
- 水平方向 **79.9%** 集中在左右边缘（后方）
- 前方（图像中心）仅 **20%** 的点

### 原因
1. **Equirectangular 投影特性**：
   - 图像中心 → agent 正前方 (phi ≈ 0)
   - 图像左右边缘 → agent 后方 (phi ≈ ±π)

2. **轨迹特性**：
   - R2R 数据：agent 沿路径向前走
   - **历史热力图**：走过的点 → 后方 → 左右边缘
   - **未来热力图**：将去的点 → 前方 → 中心（但未来路径通常较短）

3. **高度限制**：
   - 室内导航，agent 高度固定 (~1.5m)
   - 垂直方向变化极小

---

## 优化方案

### 方案 1: 区域感知损失 (Regional Focal Loss) ⭐推荐

**思路**：对稀疏区域（前方、上下）给予更高权重

```python
def regional_focal_loss(pred, gt, alpha_center=2.0, alpha_top_bottom=1.5):
    """
    区域感知损失：对中心（前方）和上下区域增加权重
    """
    H, W = gt.shape[-2:]
    
    # 创建区域权重图
    weight_map = torch.ones_like(gt)
    
    # 中心区域 (前方) 权重增加
    center_start = W // 3
    center_end = 2 * W // 3
    weight_map[..., :, center_start:center_end] *= alpha_center
    
    # 上下区域权重增加
    top_end = H // 3
    bottom_start = 2 * H // 3
    weight_map[..., :top_end, :] *= alpha_top_bottom
    weight_map[..., bottom_start:, :] *= alpha_top_bottom
    
    loss = F.mse_loss(pred, gt, reduction='none')
    weighted_loss = (loss * weight_map).mean()
    
    return weighted_loss
```

**优点**：
- 实现简单，无需修改数据
- 提升模型对稀疏区域的敏感度

---

### 方案 2: 热力图数据增强 (Circular Augmentation)

**思路**：利用全景图的环形连续性进行增强

```python
def circular_augment(heatmap, rgb, shift_range=0.3):
    """
    环形平移增强：随机水平平移全景图和热力图
    等价于 agent 旋转
    """
    W = heatmap.shape[-1]
    shift = int(random.uniform(-shift_range, shift_range) * W)
    
    # 环形平移（全景图左右是连续的）
    heatmap_aug = torch.roll(heatmap, shifts=shift, dims=-1)
    rgb_aug = torch.roll(rgb, shifts=shift, dims=-1)
    
    return heatmap_aug, rgb_aug
```

**优点**：
- 增加数据多样性
- 模拟 agent 不同朝向

**注意**：
- 需要同时平移 RGB 和热力图
- 动作/轨迹也需要相应调整（旋转 yaw）

---

### 方案 3: 双热力图策略 (已实现)

**当前设计**：
- `history_heatmap`: 历史轨迹 → 主要在后方
- `future_heatmap`: 未来轨迹 → 主要在前方

**建议改进**：
- 增加 `future_heatmap` 的训练权重
- 对前方区域给予更多关注

---

### 方案 4: 坐标归一化策略

**问题**：Equirectangular 投影的两极有严重畸变

**解决**：在损失计算时考虑投影畸变

```python
def equirect_weighted_loss(pred, gt):
    """
    考虑 equirectangular 畸变的加权损失
    极点区域（上下）面积膨胀，应降低权重
    """
    H, W = gt.shape[-2:]
    
    # 创建纬度权重（赤道=1，极点=cos(lat)）
    v = torch.linspace(0, 1, H, device=gt.device)
    lat = (0.5 - v) * math.pi  # [-pi/2, pi/2]
    lat_weight = torch.cos(lat).view(H, 1)
    
    loss = F.mse_loss(pred, gt, reduction='none')
    weighted_loss = (loss * lat_weight).mean()
    
    return weighted_loss
```

---

### 方案 5: 采样策略优化

**问题**：历史帧远多于未来帧，导致后方点占比过高

**建议**：
1. **限制历史帧数量**：仅使用最近 N 帧历史
2. **增加未来权重**：训练时增加 future_heatmap 的损失权重
3. **平衡采样**：采样时优先选择有明显前方目标的帧

---

## 推荐实施优先级

| 优先级 | 方案 | 实现难度 | 预期效果 |
|--------|------|----------|----------|
| ⭐⭐⭐ | 1. 区域感知损失 | 低 | 高 |
| ⭐⭐ | 2. 环形平移增强 | 中 | 中 |
| ⭐⭐ | 3. 调整 future_heatmap 权重 | 低 | 中 |
| ⭐ | 4. Equirect 畸变校正 | 中 | 低 |
| ⭐ | 5. 采样策略优化 | 高 | 中 |

---

## 实施建议

### 立即可做（改配置）
1. 增加 `future_heatmap_weight` 配置项
2. 调整 `focal_alpha` 对中心区域加权

### 短期改进（改代码）
1. 实现区域感知损失
2. 添加环形平移增强

### 长期优化（改数据）
1. 数据采集时增加多朝向采样
2. 考虑使用 cubemap 替代 equirectangular

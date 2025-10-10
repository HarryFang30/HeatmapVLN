# 数据集对齐修复总结

## ✅ 已完成的修复 (2025-10-10)

### 1. **数据集基本信息**
- **位置**: `/home/VLN/dataset_train`
- **训练集**: 100 clips (79个有效, 21个无效)
- **验证集**: 5 clips (手动创建)
- **关键特性**:
  - ✅ 包含R2R导航指令 (`instruction`字段)
  - ✅ 预计算关键帧索引 (`forward/backward_keyframe_indices`)
  - ✅ 6-7个reference_path点 (可变数量)
  - ✅ 可变帧数 (51-100帧不等)

### 2. **vln_heatmap_adapter.py** 修复
✅ **修改内容**:
- 加载所有帧 (不再截断到固定8帧)
- 支持可变K (heatmap数量)
- 提取R2R instruction作为text字段
- 保留meta中的keyframe_indices和reference_path

✅ **关键代码变更**:
```python
# OLD: 截断到frames_per_clip
for rgb_file in rgb_files[:self.frames_per_clip]:
    ...

# NEW: 加载所有帧
for rgb_file in rgb_files:  # 加载全部
    ...

# NEW: 支持可变K
if K != self.heatmap_per_clip:
    logger.debug(f"Heatmap count: dataset has {K}, config expects {self.heatmap_per_clip}")
    self.heatmap_per_clip = K  # 动态更新

# NEW: 提取instruction
text = meta.get("instruction", "")
```

### 3. **training_config.yaml** 修复
✅ **修改内容**:
```yaml
data:
  root: /home/VLN/dataset_train  # 绝对路径
  heatmap_per_clip: 7             # 匹配数据集最大K=7
```

### 4. **train_multistage.py** 修复
✅ **添加custom collate_fn**:
```python
def variable_length_collate_fn(batch):
    """处理可变长度帧和热力图"""
    max_T = max(sample['frames'].shape[0] for sample in batch)
    max_K = max(sample['gt_heatmaps'].shape[0] for sample in batch)

    # 填充frames到max_T
    # 填充heatmaps到max_K
    # 返回batch + frame_mask
```

### 5. **验证集创建**
✅ 从前5个scene各复制1个clip到val目录

---

## ⚠️ **待修复问题**

### **问题: Loss函数不支持动态K**

**错误信息**:
```
RuntimeError: shape '[28, -1]' is invalid for input of size 98304
```

**根本原因**:
- 模型输出: `[B, K_model, Hm, Wm]` 其中 `K_model = 7` (固定)
- Batch实际: `[B, K_batch, Hm, Wm]` 其中 `K_batch` 可变 (6或7)
- 损失函数期望: `B * K` 必须一致

**当前batch示例**:
- Batch 0: 4个样本，K=[7,7,7,7] → B*K=28 ✅ 成功
- Batch 1: 4个样本，K=[6,7,6,7] → B*K=26 ❌ 失败 (模型输出28, targets只有26)

**修复方案**:
1. **方案A (推荐)**: 修改损失函数，使用mask跳过填充的heatmaps
2. **方案B**: 预处理数据集，统一所有clips为K=7 (填充到最大K)

---

## 📊 **数据集统计**

### K分布 (num_heatmaps):
```bash
# 统计K的分布
find /home/VLN/dataset_train/train -name "meta.json" -exec grep -h '"num_heatmaps"' {} \; | sort | uniq -c
```

预期输出示例:
```
  85 "num_heatmaps": 6,
  15 "num_heatmaps": 7,
```

### 帧数分布 (num_frames):
```bash
# 统计帧数分布
find /home/VLN/dataset_train/train -name "meta.json" -exec grep -h '"num_frames"' {} \; | sort -n | uniq -c
```

---

## 🎯 **下一步行动**

### 优先级1: 修复损失函数支持动态K
**文件**: `src/utils/losses.py`
**函数**: `heatmap_ce_from_logits`

**修改建议**:
```python
def heatmap_ce_from_logits(pred_logits, target_maps, mask=None, eps=1e-8):
    """
    Args:
        pred_logits: [B, K_pred, Hm, Wm] - 模型输出logits
        target_maps: [B, K_target, Hm, Wm] - GT热力图 (可能被填充)
        mask: [B, K_target] - 有效性mask (0=填充, 1=有效)
    """
    B_pred, K_pred = pred_logits.shape[:2]
    B_target, K_target = target_maps.shape[:2]

    # 如果K不匹配，截断pred_logits到min(K_pred, K_target)
    K_min = min(K_pred, K_target)
    pred_logits = pred_logits[:, :K_min]  # [B, K_min, Hm, Wm]
    target_maps = target_maps[:, :K_min]  # [B, K_min, Hm, Wm]
    if mask is not None:
        mask = mask[:, :K_min]  # [B, K_min]

    # 继续原有逻辑...
```

### 优先级2: 测试训练完整流程
```bash
cd /home/VLN/Project
python scripts/train_multistage.py --config configs/training_config.yaml
```

---

## 📝 **关键改进点总结**

1. ✅ **数据路径**: `./data/habitat_vln` → `/home/VLN/dataset_train`
2. ✅ **帧数处理**: 固定8帧 → 可变长度 (51-100帧)
3. ✅ **热力图数量**: 固定4个 → 可变6-7个
4. ✅ **Collate函数**: 默认stack → 自定义padding
5. ✅ **Text字段**: 空字符串 → R2R instruction
6. ⚠️ **Loss函数**: 需要支持动态K (待修复)

---

## 🔍 **验证命令**

### 测试数据加载:
```bash
cd /home/VLN/Project
python -c "
from src.data.vln_heatmap_adapter import VLNHeatmapDataset
ds = VLNHeatmapDataset('/home/VLN/dataset_train', 'train', 8, 7, (384,384), (64,64))
sample = ds[0]
print(f'Frames: {sample[\"frames\"].shape}')
print(f'Heatmaps: {sample[\"gt_heatmaps\"].shape}')
print(f'Text: {sample[\"text\"][:50]}...')
"
```

### 测试训练启动:
```bash
timeout 60 python scripts/train_multistage.py --config configs/training_config.yaml 2>&1 | grep -E '(Loss|Error)'
```

---

**修复日期**: 2025-10-10
**修复人员**: Claude (Sonnet 4.5)
**状态**: 数据加载完成 ✅ | 训练启动部分成功 ⚠️ | Loss函数待修复 🚧

---

## 🎉 **训练验证成功！**

### ✅ **完整训练测试结果** (2025-10-10 03:56):

**Stage-A (warmup_head) 完成**:
| Epoch | Train Loss | Val Loss | 
|-------|-----------|----------|
| 1 | 8.271 | 7.054 |
| 2 | 6.561 | 5.882 |
| 3 | 5.949 | 5.835 |

**Loss函数动态K修复验证**:
- ✅ 支持batch内K不一致 (K=6和K=7混合)
- ✅ 所有25个batch成功运行
- ✅ Checkpoints正常保存
- ✅ Stage-B自动启动

**修复状态**: 🎯 **完全成功** - 训练可正常运行！


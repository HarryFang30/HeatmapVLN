# 完整模型训练配置总结

## ✅ 已完成的工作

### 1. 内存分析
- **VGGT**: 0.47 GB (训练) - 85M参数
- **DINOv3-large**: 6.15 GB (训练) - 1.1B参数
- **Qwen2.5-VL**: 15.65 GB (冻结推理) - 7B参数
- **其他组件**: 4.28 GB (融合、转换器、激活值)
- **总计**: 约26.55 GB (阶段3最大值)

### 2. GPU分配方案（方案A - 三卡分散）
```yaml
vggt_gpu: cuda:0      # VGGT (0.47 GB)   - 剩余47.5 GB
dinov3_gpu: cuda:1    # DINOv3 (6.15 GB) - 剩余42 GB
llm_gpu: cuda:2       # Qwen + 其他 (19.93 GB) - 剩余28 GB
```

### 3. 配置文件修正
**文件**: `configs/training_config_full_model.yaml`

**关键修正**:
- ✅ 移除无效的`dinov3.model_name`和`checkpoint_path`（模型从`./models/dinov3`自动加载）
- ✅ LLM路径修正为`./models/qwen_2.5_vl`
- ✅ Gradient checkpointing禁用（VGGT不支持）
- ✅ 三卡GPU分配

### 4. 训练脚本修正
**文件**: `scripts/train_full_model.py`

**关键修正**:
- ✅ 添加`variable_length_collate_fn`函数（处理变长序列）
- ✅ 修正heatmap_size配置读取（从`cfg['data']['init_hm_size']`）
- ✅ 添加dict→tensor转换逻辑（处理模型输出）
- ✅ 启用gradient checkpointing配置传递

## ⚠️  当前问题

### 问题：热力图尺寸不匹配
**错误信息**:
```
RuntimeError: The size of tensor a (4096) must match the size of tensor b (802816)
```

**分析**:
- 预测热力图: [B, K, 896, 128] → 802816 = 7 × 896 × 128
- GT热力图: [B, K, 64, 64] → 4096 = 64 × 64
- **根本原因**: `Spatial MLLMPipeline`的`heatmap_converter`没有使用配置中的`heatmap_size`参数

**可能的原因**:
1. `heatmap_converter`在初始化时没有正确接收`heatmap_size`参数
2. 输出尺寸被硬编码在`LLMToHeatmapConverter`中
3. Config的`heatmap_size`没有正确传递到converter

## 📋 待修复

### 修复步骤

#### 1. 检查`SpatialMLLMPipeline`的heatmap_converter初始化
```python
# 文件: src/models/spatial_mllm_compat.py
# 查找heatmap_converter初始化代码，确认heatmap_size是否被传递
```

#### 2. 检查`LLMToHeatmapConverter`实现
```python
# 文件: src/models/heatmap/converter.py
# 确认converter是否使用heatmap_size参数生成输出
```

#### 3. 验证修复
```bash
cd /home/VLN/Project
python scripts/debug_heatmap_size.py
```

预期输出应该是:
```
Stacked shape: [1, 7, 64, 64]
Expected shape: [1, 7, 64, 64]
```

#### 4. 重新启动训练
```bash
export CUDA_VISIBLE_DEVICES=0,1,2
cd /home/VLN/Project
python scripts/train_full_model.py --config configs/training_config_full_model.yaml
```

## 📊 训练配置总结

### 三阶段训练
1. **Stage 1**: 预热热力图头 (64×64, 5 epochs)
   - 冻结: VGGT, DINOv3, Fusion, LLM
   - 训练: heatmap_converter only
   - 学习率: 1e-3

2. **Stage 2**: 微调融合模块 (64×64, 8 epochs)
   - 冻结: VGGT, DINOv3, LLM
   - 训练: heatmap_converter, feature_fusion, llm_projector
   - 学习率: 1e-3 (head), 5e-4 (fusion)

3. **Stage 3**: 微调编码器 (128×128, 10 epochs)
   - 冻结: LLM only
   - 训练: 所有组件 (VGGT, DINOv3, fusion, converter)
   - 学习率: 1e-3 (head), 5e-4 (fusion), 1e-5 (encoders)

### 关键超参数
- Batch size: 1
- Gradient accumulation: 8 (有效batch=8)
- Optimizer: AdamW
- Weight decay: 1e-2
- Gradient clip: 1.0
- Mixed precision: BFloat16

## 🔍 调试命令

### 检查GPU状态
```bash
nvidia-smi
```

### 检查训练日志
```bash
tail -f /home/VLN/Project/outputs_full_model/train.log
```

### 检查模型参数
```python
# 总参数: 8,077,792,829 (8.08B)
# 阶段1可训练: 80,017,553 (0.99%)
# 阶段2可训练: ~100M (1.2%)
# 阶段3可训练: ~1.2B (14.8%)
```

## 🎯 下一步行动

1. ✅ 分析内存需求（已完成）
2. ✅ 设计GPU分配方案（已完成）
3. ✅ 修正配置文件（已完成）
4. ✅ 修正训练脚本（已完成）
5. ⏳ **修复热力图尺寸不匹配**（当前任务）
6. ⏳ 完成完整模型训练
7. ⏳ 评估与验证

---

**状态**: 内存配置完成 ✅ | 训练脚本就绪 ✅ | 尺寸Bug待修 ⚠️
**日期**: 2025-10-10

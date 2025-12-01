# 完整模型训练 - 所有问题已修复 ✅

## 🎉 成功解决的所有问题

### 1. 热力图尺寸不匹配 ✅
**问题**: 模型输出896×128，但GT是64×64
**根本原因**: `heatmap_converter`输出尺寸取决于输入spatial tokens的维度
**解决方案**: 在`spatial_mllm_compat.py:669-680`添加动态resize
```python
# Resize to target heatmap size if needed
target_h, target_w = self.config.heatmap_size
if current_h != target_h or current_w != target_w:
    frame_heatmap = torch.nn.functional.interpolate(
        frame_heatmap, size=(target_h, target_w),
        mode='bilinear', align_corners=False
    )
```

### 2. 设备不匹配 ✅
**问题**: 预测在cuda:2，GT在cuda:0
**解决方案**: 在`train_full_model.py:334-336`添加设备对齐
```python
if pred_heatmaps.device != gt_heatmaps.device:
    pred_heatmaps = pred_heatmaps.to(gt_heatmaps.device)
```

### 3. GPU 0内存溢出 ✅
**问题**: Feature fusion在GPU 0导致37GB+内存占用
**解决方案**:
- `spatial_mllm_compat.py:164-166`: 将fusion移到GPU 2
- `spatial_mllm_compat.py:573-578`: 自动将特征移到fusion设备

```python
# Feature fusion on llm_gpu
fusion_device = torch.device(config.llm_gpu if config.use_multi_gpu else config.device)
self.feature_fusion = self._create_feature_fusion_module().to(device=fusion_device)

# Auto move features
fusion_device = next(self.feature_fusion.parameters()).device
vggt_features = vggt_features.to(fusion_device)
dinov3_features = dinov3_features.to(fusion_device)
```

### 4. Dict输出提取 ✅
**问题**: 模型返回dict而非tensor
**解决方案**: `train_full_model.py:326-332`添加dict→tensor转换
```python
if isinstance(heatmaps_output, dict):
    sorted_keys = sorted(heatmaps_output.keys())
    pred_heatmaps = torch.stack([heatmaps_output[k] for k in sorted_keys], dim=1)
```

### 5. 配置文件清理 ✅
- 移除无效的`dinov3.model_name`配置
- 修正LLM路径为`./models/qwen_2.5_vl`
- 禁用gradient checkpointing（VGGT不支持）

## 📊 最终GPU分配（方案A）

```
GPU 0: VGGT (0.47 GB训练)              剩余: ~47 GB
GPU 1: DINOv3 (6.15 GB训练)            剩余: ~42 GB
GPU 2: Qwen + Fusion + Converter       剩余: ~28 GB
       (15.65 + 4.28 = 19.93 GB)
GPU 3: 备用
```

## 🚀 启动训练

### 方法1：使用启动脚本（推荐）
```bash
cd /home/VLN/Project
./scripts/start_full_training.sh
```

### 方法2：直接命令
```bash
export CUDA_VISIBLE_DEVICES=0,1,2
cd /home/VLN/Project
nohup python scripts/train_full_model.py \
  --config configs/training_config_full_model.yaml \
  > outputs_full_model/training_run.log 2>&1 &
```

### 监控训练
```bash
# 查看训练日志
tail -f outputs_full_model/training_run.log

# 监控GPU状态
watch -n 1 nvidia-smi

# 检查检查点
ls -lh outputs_full_model/warmup_heatmap_head_64/
```

## 📈 预期训练进度

### Stage 1: 预热热力图头 (64×64)
- Epochs: 5
- 可训练参数: 80M (0.99%)
- 批次时间: ~48秒/batch
- 总时间: ~4小时

### Stage 2: 微调融合模块 (64×64)
- Epochs: 8
- 可训练参数: ~100M (1.2%)
- 总时间: ~6.5小时

### Stage 3: 微调编码器 (128×128)
- Epochs: 10
- 可训练参数: ~1.2B (14.8%)
- 总时间: ~10小时

**总预计时间**: ~20.5小时

## ✅ 验证清单

- [x] 配置文件正确 (`training_config_full_model.yaml`)
- [x] 模型正确加载 (VGGT + DINOv3 + Qwen)
- [x] 热力图尺寸匹配 (64×64)
- [x] 设备分配正确 (3-GPU split)
- [x] 内存优化完成 (fusion on GPU 2)
- [x] 第一个batch成功训练
- [x] 训练脚本就绪

## 🎯 关键文件

- 配置: [configs/training_config_full_model.yaml](configs/training_config_full_model.yaml)
- 训练脚本: [scripts/train_full_model.py](scripts/train_full_model.py)
- 启动脚本: [scripts/start_full_training.sh](scripts/start_full_training.sh)
- 模型文件: [src/models/spatial_mllm_compat.py](src/models/spatial_mllm_compat.py)

## 📝 修改的文件

1. `src/models/spatial_mllm_compat.py`
   - Line 165-166: Feature fusion移到GPU 2
   - Line 573-578: 自动设备对齐
   - Line 669-680: 热力图动态resize

2. `scripts/train_full_model.py`
   - Line 326-332: Dict→Tensor转换
   - Line 334-336: 设备对齐
   - Line 406-408: 验证函数设备对齐

3. `configs/training_config_full_model.yaml`
   - 清理无效DINOv3配置
   - 修正LLM路径
   - 禁用gradient checkpointing

---

**状态**: 所有问题已修复 ✅ | 训练就绪 🚀 | 可以开始训练！

**日期**: 2025-10-10
**预期完成**: 2025-10-11 (20.5小时后)

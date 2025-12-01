# Frame-by-Frame GT vs Prediction Visualization

## 概述

已成功实现逐帧GT vs 预测热力图对比可视化功能，支持三种显示模式。

## 新增文件

```
Project/
├── src/utils/
│   ├── frame_vis_utils.py      # 插值和叠加核心工具
│   ├── visualization.py         # 网格生成和缩略图
│   └── html_template.py         # HTML索引生成
└── scripts/
    └── evaluate.py (已修改)     # 添加新的CLI参数和可视化分支
```

## 使用方法

### 1. 基本用法（逐帧双叠加模式）

```bash
python /root/VLN/Project/scripts/evaluate.py \
  --config /path/to/training_config.yaml \
  --checkpoint /path/to/checkpoint.pth \
  --split val \
  --save_vis \
  --num-vis 50 \
  --vis-mode frame_by_frame \
  --overlay-mode dual
```

**输出**：
- 4列网格：RGB原图 | GT叠加(历史=青色+未来=红色) | 预测叠加 | 差异图
- HTML索引页面：`{out_dir}/eval_vis/index.html`

### 2. 分离显示历史和未来

```bash
python scripts/evaluate.py \
  --config configs/training_config.yaml \
  --checkpoint checkpoint.pth \
  --split val \
  --save_vis \
  --num-vis 50 \
  --vis-mode frame_by_frame \
  --overlay-mode separate  # 生成两个独立网格
```

**输出**：
- `grid_comparison_history.png`：历史热力图4列网格
- `grid_comparison_future.png`：未来热力图4列网格

### 3. 完全分离模式（7列超宽网格）

```bash
python scripts/evaluate.py \
  --config configs/training_config.yaml \
  --checkpoint checkpoint.pth \
  --split val \
  --save_vis \
  --num-vis 20 \
  --vis-mode frame_by_frame \
  --overlay-mode full-separate
```

**输出**：
- 7列网格：RGB | 历史GT | 历史预测 | 历史差异 | 未来GT | 未来预测 | 未来差异

### 4. 向后兼容（聚合模式）

```bash
python scripts/evaluate.py \
  --config configs/training_config.yaml \
  --checkpoint checkpoint.pth \
  --split val \
  --save_vis \
  --num-vis 20 \
  --vis-mode aggregated  # 保持原有3×3网格
```

### 5. 同时生成两种模式

```bash
python scripts/evaluate.py \
  --config configs/training_config.yaml \
  --checkpoint checkpoint.pth \
  --split val \
  --save_vis \
  --vis-mode both  # 生成聚合 + 逐帧两种可视化
```

## 所有新增CLI参数

| 参数 | 默认值 | 选项 | 说明 |
|------|--------|------|------|
| `--vis-mode` | `aggregated` | `frame_by_frame`, `aggregated`, `both` | 可视化模式 |
| `--overlay-mode` | `dual` | `dual`, `separate`, `full-separate` | 叠加模式 |
| `--max-frames-per-vis` | `16` | 整数 | 每个网格显示的最大帧数 |
| `--no-html` | False | flag | 禁用HTML索引生成 |
| `--interpolation-method` | `linear` | `linear`, `nearest`, `cubic` | 插值方法 |
| `--overlay-alpha` | `0.5` | 0-1 | 热力图透明度 |
| `--heatmap-threshold` | `0.05` | 0-1 | 显示的最小热力图值 |

## 输出结构

```
{out_dir}/eval_vis/
├── index.html                              # 主索引页
├── summary_metrics.json                    # 聚合指标
└── samples/
    ├── sample_0000/
    │   ├── grid_comparison.png             # dual模式：4列×16行
    │   ├── grid_comparison_history.png     # separate模式：历史4列×16行
    │   ├── grid_comparison_future.png      # separate模式：未来4列×16行
    │   └── thumbnail.png                   # 首帧叠加缩略图
    └── sample_0001/
        └── ...
```

## 关键特性

### 1. 时间对齐插值
- **问题**：数据集提供T=16帧GT，模型输出K个关键帧
- **解决**：使用scipy插值将K关键帧扩展到T帧
- **方法**：线性/最近邻/三次插值（可选）

### 2. 双色图叠加
- **历史热力图**：WINTER颜色映射（青色）
- **未来热力图**：HOT颜色映射（红黄色）
- **视觉直观**：冷色=过去，暖色=未来

### 3. HTML索引
- 响应式3列网格布局
- 缩略图预览
- 导航指令显示
- 指标汇总（MAE、有效帧数等）
- 可点击链接查看完整网格

## 技术细节

### 插值算法 (`frame_vis_utils.py`)

```python
def interpolate_keyframe_predictions(
    pred_keyframes: torch.Tensor,  # [K, H, W]
    keyframe_indices: np.ndarray,   # [K]
    total_frames: int,              # T
    method: str = 'linear'
) -> torch.Tensor:                  # [T, H, W]
```

- 对每个像素的时间序列进行1D插值
- 插值后重归一化保持概率分布特性
- 边界外推使用extrapolate模式

### 网格生成 (`visualization.py`)

```python
def create_comparison_grid(
    frames, gt_hm_hist, gt_hm_fut,
    pred_hm_hist, pred_hm_fut,
    gt_val_hist, gt_val_fut,
    save_dir, overlay_mode, ...
) -> Dict[str, str]
```

- 根据`overlay_mode`生成不同布局
- 返回生成的文件路径字典
- 支持多种colormaps和透明度

## 测试

### 快速测试（5个样本）

```bash
cd /root/VLN/Project

python scripts/evaluate.py \
  --config configs/training_config_full_model.yaml \
  --checkpoint /path/to/checkpoint.pth \
  --split val \
  --save_vis \
  --num-vis 5 \
  --vis-mode frame_by_frame \
  --overlay-mode dual
```

### 验证输出

```bash
# 检查生成的文件
ls -R /root/autodl-tmp/vln_dual_head_outputs/eval_vis/

# 应该看到：
# - index.html
# - summary_metrics.json
# - samples/sample_0000/grid_comparison.png
# - samples/sample_0000/thumbnail.png
# - ...
```

### 浏览HTML索引

在浏览器中打开：
```
/root/autodl-tmp/vln_dual_head_outputs/eval_vis/index.html
```

## 依赖项

- **已验证**：scipy >= 1.5.3（已安装）
- **已有**：numpy, torch, cv2, matplotlib

## 向后兼容性

- 默认`--vis-mode=aggregated`保持原有行为
- 原有的`visualize_sample()`函数保留
- 所有新功能通过显式参数启用

## 已知限制

1. **关键帧索引假设**：当前假设关键帧均匀分布，未来可通过修改模型暴露实际索引
2. **插值伪影**：线性插值可能在不连续区域产生伪影，可通过`--interpolation-method nearest`缓解
3. **内存占用**：大量样本的完整网格会占用较多磁盘空间（每个样本约1-12MB）

## 故障排查

### 问题1：ImportError for scipy
```bash
pip install scipy>=1.10.0
```

### 问题2：可视化网格过大
```bash
# 减少显示帧数
--max-frames-per-vis 8

# 或限制样本数量
--num-vis 20
```

### 问题3：HTML索引不显示缩略图
- 检查相对路径是否正确
- 确保`thumbnail.png`已生成
- 浏览器控制台查看错误信息

## 下一步优化

1. **模型集成**：修改`SpatialMLLMPipeline`暴露实际关键帧索引
2. **交互式HTML**：添加过滤/排序功能
3. **视频生成**：将帧序列导出为MP4动画
4. **并行化**：使用multiprocessing加速网格生成

---

**状态**：✅ 核心功能已实现并可用
**测试**：待用户在实际数据集上验证

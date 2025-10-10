# vln_heatmap_adapter.py 期望数据格式 vs 当前数据集差异说明

## 📋 当前数据集结构（实际拥有）

```
/home/VLN/dataset_train/
├── train/
│   ├── {scene_id}/                    # 例如: 17DRP5sb8fy, 1LXtFkjw3qL
│   │   ├── clip_XXXXXX/               # 例如: clip_000021, clip_000082
│   │   │   ├── rgb/
│   │   │   │   ├── 000000.png        # RGB图像序列
│   │   │   │   ├── 000001.png
│   │   │   │   └── ... (N帧)
│   │   │   ├── poses.json             # 相机位姿数组 [N × 4×4]
│   │   │   ├── intrinsics.json        # 相机内参 {fx, fy, cx, cy, K, width, height, hfov}
│   │   │   └── meta.json              # VLN元数据
│   │   └── clip_YYYYYY/
│   └── {other_scenes}/
├── train_index.json                    # 全局索引
├── collection_stats.json               # 统计信息
└── progress.json                       # 采集进度
```

### 现有文件详细说明

#### 1. `rgb/*.png` ✅
- **格式**: PNG图像
- **分辨率**: 224×224
- **通道**: 3 (RGB)
- **数量**: 每个clip 26-100帧不等
- **命名**: `000000.png`, `000001.png`, ...
- **状态**: ✅ **已有，符合要求**

#### 2. `poses.json` ✅
```json
[
  [
    [R11, R12, R13, tx],    // 4×4变换矩阵
    [R21, R22, R23, ty],
    [R31, R32, R33, tz],
    [0.0, 0.0, 0.0, 1.0]
  ],
  [...],  // 第2帧位姿
  [...]   // 第N帧位姿
]
```
- **格式**: JSON数组，每个元素为4×4矩阵（列表的列表）
- **内容**: 世界坐标系到相机坐标系的变换 T_w_c
- **数量**: N个（与RGB帧数相同）
- **状态**: ✅ **已有，符合要求**

#### 3. `intrinsics.json` ✅
```json
{
  "fx": 112.0,
  "fy": 112.0,
  "cx": 112.0,
  "cy": 112.0,
  "K": [
    [112.0, 0.0, 112.0],
    [0.0, 112.0, 112.0],
    [0.0, 0.0, 1.0]
  ],
  "width": 224,
  "height": 224,
  "hfov": 90
}
```
- **格式**: JSON对象
- **内容**: 针孔相机模型参数
- **状态**: ✅ **已有，符合要求**

#### 4. `meta.json` ✅
```json
{
  "episode_id": "2889",
  "trajectory_id": "1901",
  "scene_id": "17DRP5sb8fy",
  "instruction": "Walk forward into the bathroom. Wait near the sink.",
  "sampling_strategy": "bidirectional_walk",
  "num_frames": 76,
  "forward_segment": {...},
  "backward_segment": {...},
  "reference_path": [              // 3D路径点
    [-2.209, 0.072, 0.492],
    [-0.915, 0.072, 0.412],
    [1.441, 0.072, 0.394],
    [2.414, 0.072, 0.397],
    [3.507, 0.072, 0.492]
  ],
  "forward_keyframe_indices": [0, 0, 15, 20, 25],
  "forward_keyframe_distances": [...],
  "backward_keyframe_indices": [26, 38, 42, 57, 75],
  "backward_keyframe_distances": [...]
}
```
- **格式**: JSON对象
- **关键字段**:
  - `reference_path`: R2R数据集的3D关键路径点
  - `forward/backward_keyframe_indices`: 每个reference点对应的帧索引
- **状态**: ✅ **已有，符合要求**

---

## ❌ 训练脚本期望但**缺失**的数据

### 5. `heatmaps.npy` ❌ **缺失 - 核心问题**

**期望格式**:
```python
# NumPy数组
shape: (K, H, W)
dtype: float32
range: [0, 1]
normalization: 每个heatmap[k]的sum应为1.0（概率分布）
```

**详细说明**:
- **K**: 热力图数量（训练配置中为4，但实际应该等于`len(reference_path)`）
- **H, W**: 热力图分辨率（64×64, 128×128, 或224×224）
- **内容**: 每个热力图表示一个关键帧（keyframe）在当前观察视角中的**空间概率分布**
- **物理意义**: `heatmaps[k, y, x]` = 关键帧k在像素(x,y)位置出现的概率

**生成方法（需要实现）**:

#### 方法A: 基于reference_path的几何投影（推荐）

```python
# 对每个reference_path点生成热力图
for k, ref_point_3d in enumerate(reference_path):
    # 1. 将3D点投影到每一帧的2D图像平面
    for frame_idx in range(num_frames):
        T_w_c = poses[frame_idx]          # 世界→相机变换
        K = intrinsics['K']                # 相机内参

        # 投影: 世界坐标 → 相机坐标 → 图像平面
        p_cam = T_w_c @ [ref_point_3d, 1]  # 变换到相机坐标系
        p_img = K @ p_cam[:3]               # 投影到图像平面
        u, v = p_img[0]/p_img[2], p_img[1]/p_img[2]  # 归一化

        # 2. 在(u, v)位置生成2D高斯分布
        heatmap[k] = gaussian_2d(center=(u, v), sigma=2.0, size=(H, W))

        # 3. 归一化为概率分布
        heatmap[k] /= heatmap[k].sum()
```

**关键参数**:
- `sigma`: 高斯分布标准差（像素单位），控制热力图"宽度"
  - 推荐值: `sigma = 2.0~3.0` (对于64×64)
  - 推荐值: `sigma = 4.0~6.0` (对于128×128)
- `lookback`: 使用哪些帧生成热力图（建议使用meta.json中的`forward_keyframe_indices`）

#### 方法B: 简化版（仅用于快速测试）

```python
# 直接在keyframe_indices对应的帧中心位置生成高斯分布
for k, frame_idx in enume
rate(forward_keyframe_indices):
    # 在图像中心生成固定高斯分布
    center = (H//2, W//2)
    heatmap[k] = gaussian_2d(center=center, sigma=3.0, size=(H, W))
    heatmap[k] /= heatmap[k].sum()
```

**保存格式**:
```python
import numpy as np
np.save('heatmaps.npy', heatmaps)  # shape: (K, H, W), dtype: float32
```

---

### 6. `mask.npy` ❌ **缺失 - 必需**

**期望格式**:
```python
# NumPy数组
shape: (K,)
dtype: float32
values: 0.0 或 1.0
```

**详细说明**:
- **K**: 与heatmaps数量相同
- **内容**: 指示每个热力图是否有效
  - `mask[k] = 1.0`: 热力图k是有效的（应该用于训练）
  - `mask[k] = 0.0`: 热力图k无效（跳过，不参与损失计算）

**生成逻辑**:
```python
mask = np.ones(K, dtype=np.float32)  # 默认全部有效

# 检查条件：如果某个reference_path点投影到图像外，标记为无效
for k, ref_point_3d in enumerate(reference_path):
    T_w_c = poses[some_frame_idx]
    K_mat = intrinsics['K']

    # 投影
    p_cam = T_w_c @ [ref_point_3d, 1]
    p_img = K_mat @ p_cam[:3]
    u, v = p_img[0]/p_img[2], p_img[1]/p_img[2]

    # 检查是否在图像内
    if not (0 <= u < width and 0 <= v < height):
        mask[k] = 0.0  # 标记为无效

    # 检查是否在相机后方
    if p_cam[2] <= 0:
        mask[k] = 0.0
```

**简化版（快速测试）**:
```python
# 所有热力图都标记为有效
mask = np.ones(K, dtype=np.float32)
```

**保存格式**:
```python
np.save('mask.npy', mask)  # shape: (K,), dtype: float32
```

---

### 7. `depth/*.npy` ⚠️ **可选（但强烈建议）**

**期望格式**:
```python
# 每帧一个文件
depth/000000.npy
depth/000001.npy
...

# NumPy数组
shape: (H, W)
dtype: float32
units: 米 (meters)
range: [0, 10.0] (典型室内场景)
```

**详细说明**:
- **用途**: 用于遮挡检测，提高热力图质量
- **物理意义**: `depth[y, x]` = 像素(x, y)处的场景深度
- **如何使用**:
  ```python
  # 检查reference_path点是否被遮挡
  ref_depth = calculate_depth_from_pose(ref_point_3d, T_w_c)
  observed_depth = depth[v, u]

  if ref_depth > observed_depth + threshold:
      # 点被遮挡，不应出现在此帧的热力图中
      heatmap[k, v, u] = 0
  ```

**如果没有depth**:
- 可以跳过遮挡检测
- 生成的热力图可能包含一些"透视"错误（看到墙后的点）
- **对训练影响**: 中等（会有噪声，但模型可以学习）

---

## 📐 完整数据生成流程（推荐实现）

### 输入数据（已有）
```
clip_000021/
├── rgb/*.png              (N张)
├── poses.json             (N个4×4矩阵)
├── intrinsics.json        (相机参数)
└── meta.json              (包含reference_path和keyframe_indices)
```

### 处理流程

```python
# Step 1: 加载数据
rgb_frames = load_rgb_frames('rgb/')                    # (N, 224, 224, 3)
poses = load_poses('poses.json')                        # (N, 4, 4)
K_mat = load_intrinsics('intrinsics.json')['K']        # (3, 3)
meta = load_meta('meta.json')
reference_path = meta['reference_path']                 # (K_ref, 3)
forward_kf_indices = meta['forward_keyframe_indices']   # (K_ref,)

# Step 2: 确定关键帧
# 方案A: 使用meta中的forward_keyframe_indices
keyframe_indices = forward_kf_indices

# 方案B: 自己重新采样（如果想改变K）
# keyframe_indices = uniform_sample(N, K=4)

K = len(keyframe_indices)

# Step 3: 为每个关键帧生成热力图
heatmaps = np.zeros((K, 64, 64), dtype=np.float32)
mask = np.ones(K, dtype=np.float32)

for k in range(K):
    ref_point = reference_path[k]                      # 3D点 [x, y, z]
    kf_idx = keyframe_indices[k]                       # 对应的帧索引
    T_w_c = poses[kf_idx]                              # 该帧的相机位姿

    # 投影3D点到2D图像
    p_world = np.array([ref_point[0], ref_point[1], ref_point[2], 1.0])
    p_cam = T_w_c @ p_world                            # 变换到相机坐标系

    # 检查是否在相机前方
    if p_cam[2] <= 0:
        mask[k] = 0.0
        continue

    # 投影到图像平面
    p_img = K_mat @ p_cam[:3]
    u = p_img[0] / p_img[2]
    v = p_img[1] / p_img[2]

    # 检查是否在图像内
    if not (0 <= u < 224 and 0 <= v < 224):
        mask[k] = 0.0
        continue

    # 生成2D高斯热力图
    heatmap = generate_gaussian_heatmap(
        center=(u, v),
        sigma=2.0,
        size=(64, 64),
        original_size=(224, 224)
    )

    # 归一化
    if heatmap.sum() > 0:
        heatmap /= heatmap.sum()

    heatmaps[k] = heatmap

# Step 4: 保存
np.save('heatmaps.npy', heatmaps)  # (K, 64, 64), float32
np.save('mask.npy', mask)          # (K,), float32
```

### 输出数据（目标）
```
clip_000021/
├── rgb/*.png              ✅ (已有)
├── poses.json             ✅ (已有)
├── intrinsics.json        ✅ (已有)
├── meta.json              ✅ (已有)
├── heatmaps.npy           ❌ (需要生成) - (K, 64, 64), float32
└── mask.npy               ❌ (需要生成) - (K,), float32
```

---

## 🛠️ 辅助函数参考实现

### 生成2D高斯热力图

```python
def generate_gaussian_heatmap(center, sigma, size, original_size=None):
    """
    生成2D高斯热力图

    Args:
        center: (u, v) - 中心坐标（原始图像坐标系）
        sigma: float - 高斯标准差（像素单位）
        size: (H, W) - 输出热力图尺寸
        original_size: (H_orig, W_orig) - 原始图像尺寸（用于坐标缩放）

    Returns:
        heatmap: (H, W) - 归一化的概率分布
    """
    H, W = size
    u, v = center

    # 如果需要缩放坐标
    if original_size is not None:
        H_orig, W_orig = original_size
        u = u * W / W_orig
        v = v * H / H_orig
        sigma = sigma * W / W_orig  # 缩放sigma

    # 生成网格
    x = np.arange(0, W, dtype=np.float32)
    y = np.arange(0, H, dtype=np.float32)
    xx, yy = np.meshgrid(x, y)

    # 计算高斯分布
    heatmap = np.exp(-((xx - u)**2 + (yy - v)**2) / (2 * sigma**2))

    # 归一化
    if heatmap.sum() > 0:
        heatmap /= heatmap.sum()

    return heatmap
```

---

## 📊 数据规格总结

| 文件 | 状态 | 格式 | Shape | DType | 用途 |
|------|------|------|-------|-------|------|
| `rgb/*.png` | ✅ 已有 | PNG | (224, 224, 3) | uint8 | 输入图像 |
| `poses.json` | ✅ 已有 | JSON | (N, 4, 4) | float | 相机位姿 |
| `intrinsics.json` | ✅ 已有 | JSON | - | - | 相机参数 |
| `meta.json` | ✅ 已有 | JSON | - | - | VLN元数据 |
| **`heatmaps.npy`** | ❌ **缺失** | NPY | **(K, H, W)** | **float32** | **监督信号** |
| **`mask.npy`** | ❌ **缺失** | NPY | **(K,)** | **float32** | **有效性掩码** |
| `depth/*.npy` | ⚠️ 可选 | NPY | (224, 224) | float32 | 遮挡检测 |

**关键参数**:
- **N**: 每个clip的帧数（26~100不等）
- **K**: 热力图数量（= `len(reference_path)`，通常5~10个）
- **H, W**: 热力图分辨率（64×64, 128×128, 或224×224）

---

## 🎯 快速验证清单

生成数据后，请验证以下内容：

### 1. 文件完整性
```bash
# 每个clip应该包含
ls clip_XXXXXX/
# 输出应包含: rgb/ poses.json intrinsics.json meta.json heatmaps.npy mask.npy
```

### 2. 数据shape正确性
```python
import numpy as np

heatmaps = np.load('heatmaps.npy')
mask = np.load('mask.npy')

print(f"Heatmaps shape: {heatmaps.shape}")  # 应该是 (K, H, W)
print(f"Mask shape: {mask.shape}")          # 应该是 (K,)
print(f"Heatmaps dtype: {heatmaps.dtype}")  # 应该是 float32
print(f"Mask dtype: {mask.dtype}")          # 应该是 float32
```

### 3. 数据范围正确性
```python
print(f"Heatmaps range: [{heatmaps.min()}, {heatmaps.max()}]")  # 应该在 [0, 1]
print(f"Mask values: {np.unique(mask)}")                        # 应该只有 0.0 和 1.0

# 检查归一化
for k in range(len(mask)):
    if mask[k] > 0.5:  # 有效热力图
        hm_sum = heatmaps[k].sum()
        print(f"Heatmap {k} sum: {hm_sum:.6f}")  # 应该接近 1.0
        assert abs(hm_sum - 1.0) < 1e-3, f"Heatmap {k} not normalized!"
```

### 4. 可视化检查（推荐）
```python
import matplotlib.pyplot as plt

# 显示第一个有效热力图
for k in range(len(mask)):
    if mask[k] > 0.5:
        plt.figure(figsize=(6, 6))
        plt.imshow(heatmaps[k], cmap='hot')
        plt.colorbar()
        plt.title(f'Heatmap {k} (sum={heatmaps[k].sum():.4f})')
        plt.savefig(f'heatmap_{k}_vis.png')
        break
```

---

## 💡 推荐生成策略

### 方案A: 完整几何投影（推荐，质量最高）
- ✅ 使用`reference_path` 3D坐标
- ✅ 使用`poses.json`和`intrinsics.json`进行几何投影
- ✅ 生成准确的空间分布热力图
- ⚠️ 需要实现投影逻辑（约200行代码）

### 方案B: 简化投影（快速实现）
- ✅ 使用`forward_keyframe_indices`确定关键帧
- ✅ 在关键帧的图像中心生成固定高斯分布
- ⚠️ 不考虑实际3D位置，仅用于pipeline测试

### 方案C: 基于关键帧索引的均匀分布
- ✅ 最简单实现
- ✅ 仅用于验证训练代码能否运行
- ❌ 热力图质量最低，不适合正式训练

---

## 📞 需要帮助？

如果您在生成数据时遇到问题，可以提供：
1. **示例clip路径**: 例如 `train/17DRP5sb8fy/clip_000021/`
2. **问题描述**: 遇到的具体错误或疑问
3. **生成的数据**: `heatmaps.npy`和`mask.npy`的shape和统计信息

我可以帮助：
- 提供完整的数据生成脚本
- 调试投影逻辑
- 优化热力图质量参数

---

**生成数据后，训练脚本应该可以直接使用，无需修改！**

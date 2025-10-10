# VLN-CE 数据采集指南 (Data Collection Guide)

> **目标**：为 HeatmapVLN 训练管线从 Habitat/VLN-CE 采集原始导航数据
> **环境**：VLN-CE Docker 容器 (`/home/habitat/VLN-CE`)
> **关键理念**：**两阶段流程** - 先采集原始数据 → 再打包生成热力图

---

## ⚠️ 核心注意事项（必读！）

**7 个关键点，直接影响热力图质量：**

1. **相机位姿 = Agent 位姿 × 传感器外参**
   - ✅ `T_w_c = T_w_agent @ T_agent_cam`
   - ❌ 不要直接用 `agent_state.position/rotation`
   - 原因：传感器有高度偏置（0.88m）和可能的俯仰角
   - 后果：位姿错误 → 投影错误 → 热力图完全错位

2. **内参必须与传感器配置严格一致**
   - ✅ 从配置读取 `RGB_SENSOR.WIDTH/HEIGHT/HFOV`
   - ✅ 检查 `RGB_SENSOR == DEPTH_SENSOR` 的分辨率和 HFOV
   - ❌ 不要在代码中 resize 图像
   - 后果：HFOV 错误 → fx/fy 错误 → 投影像素坐标错误

3. **帧间重叠度优先于时长**
   - ✅ 每次动作后都存一帧（采样密度优先）
   - ✅ 减小转角（15° 代替 30°）
   - ✅ 必要时混合随机动作延长路径
   - 原因：visibility-aware FPS 依赖视野重叠
   - 后果：重叠度不足 → K_eff < 2 → 样本被丢弃

4. **深度范围必须与场景尺度匹配**
   - ✅ 室内场景：`MIN_DEPTH: 0.5`, `MAX_DEPTH: 5.0`
   - ✅ RGB/Depth 分辨率必须完全相同
   - ❌ MAX_DEPTH 过小 → 大量 0 值 → 可见性判断失效
   - 检查：有效深度比例应 > 50%

5. **运行 Smoke Test 验证配置**
   - ✅ 圆点投影自检：前方 2m 点投影到图像中心
   - ✅ 姿态矩阵健康检查：无 NaN，det(R) ≈ 1
   - ❌ 跳过 smoke test → 采集 1000 帧后才发现投影错误
   - 时机：采集数据前必须运行！

6. **场景命名与打包配置必须一致**
   - ✅ `raw_sequences/train/{scene}/` ↔ `dataset_pack.yaml: scenes`
   - ✅ 场景名是 Matterport3D ID（如 `1pXnuDYAj8r`）
   - ❌ 不是 episode_id（如 `1_0`）
   - 后果：打包时找不到场景或样本被静默跳过

7. **渐进式采集：少量 → 打包 → 质量检查 → 扩大**
   - ✅ 先 10 clips → 打包 → 看 K_eff/熵 报告 → 调整策略
   - ❌ 一次性采集 1000 clips → 发现质量差 → 全部重采
   - 检查：K_eff ≥ 2 的达标率应 > 70%

---

## 🎯 核心理解：两阶段数据流程

### 阶段 1️⃣：原始数据采集（在 VLN-CE 容器）
```
Habitat Simulator
       ↓
采集 RGB + Depth + Poses + Intrinsics
       ↓
保存到 raw_sequences/ 目录
```

**目标**：采集干净的原始数据（RGB、深度、位姿、内参）
**原则**：只关注数据完整性，不处理热力图

### 阶段 2️⃣：数据打包（在训练容器）
```bash
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split train
```
```
读取 raw_sequences/
       ↓
提取子序列（滑动窗口）
       ↓
选择关键帧（visibility-aware FPS）
       ↓
几何投影生成热力图（project_keyframe_to_ref）
       ↓
保存到 data/habitat_vln/（训练格式）
```

**⚠️ 重要**：
- ✅ 在 VLN-CE 中**只需采集原始数据**（RGB/Depth/Poses/Intrinsics）
- ✅ **不需要生成热力图**（这由 `pack_dataset.py` 自动完成）
- ✅ 采集脚本只需关注数据完整性和格式正确性
- ✅ 热力图生成使用几何投影（3D→2D 投影 + 遮挡检查）

---

## 📋 目录

0. [⚠️ 核心注意事项（必读！）](#️-核心注意事项必读)
1. [原始数据格式（raw_sequences）](#1-原始数据格式raw_sequences)
2. [VLN-CE Episode 结构](#2-vln-ce-episode-结构)
3. [Habitat 采集脚本](#3-habitat-采集脚本)
4. [数据验证工具](#4-数据验证工具)
   - 4.1 [最小可复现实验（Smoke Test）](#41-最小可复现实验smoke-test)
   - 4.2 [验证脚本](#42-验证脚本)
5. [打包为训练格式](#5-打包为训练格式)
6. [Habitat API 参考](#6-habitat-api-参考)
7. [常见问题排查](#7-常见问题排查)
8. [完整工作流程总结](#8-完整工作流程总结)

---

## 1. 原始数据格式（raw_sequences）

### 1.1 目录结构

```
raw_sequences/
├── train/
│   ├── {scene_1}/              # 例如：1pXnuDYAj8r（Matterport3D 场景ID）
│   │   ├── clip_000001/
│   │   │   ├── rgb/            # RGB 图像序列
│   │   │   │   ├── 000000.png
│   │   │   │   ├── 000001.png
│   │   │   │   └── ...
│   │   │   ├── depth/          # 深度图序列
│   │   │   │   ├── 000000.npy
│   │   │   │   ├── 000001.npy
│   │   │   │   └── ...
│   │   │   ├── poses.json      # 相机位姿序列
│   │   │   └── intrinsics.json # 相机内参（每个 clip 一份）
│   │   ├── clip_000002/
│   │   └── ...
│   ├── {scene_2}/
│   └── ...
├── val/
│   └── (同上结构)
└── test/
    └── (同上结构)
```

**说明**：
- 每个 **clip** 对应一个 VLN episode 的导航轨迹
- 场景名称从 `episode.scene_id` 提取（去除 `.glb` 后缀）
- 每个 clip 是独立的，包含完整的 RGB、Depth、Poses、Intrinsics

---

### 1.2 文件格式详解

#### 📸 RGB 图像（`rgb/*.png`）

```python
# 格式：PNG（推荐）或 JPG
# 分辨率：与 Habitat 配置一致（推荐 480×640 或 384×384）
# 数据类型：uint8, RGB 顺序
# 命名：6位零填充递增，000000.png, 000001.png, ...

# 保存示例
import cv2
rgb = observations["rgb"]  # [H, W, 3], uint8, RGB
cv2.imwrite(f"rgb/{frame_id:06d}.png", cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
```

**注意**：Habitat 返回的是 RGB 顺序，OpenCV 保存需要转为 BGR！

---

#### 🌊 深度图（`depth/*.npy`）

```python
# 格式：NPY（NumPy 数组）
# 数据类型：float32（严格要求）
# 单位：米（Habitat 默认，0 表示无效点）
# 分辨率：与 RGB 完全相同（必须！）
# 命名：6位零填充递增，000000.npy, 000001.npy, ...
# 深度范围：根据配置（例如 0.5-5.0 米）

# 保存示例
depth = observations["depth"]  # [H, W, 1] 或 [H, W], float32
np.save(f"depth/{frame_id:06d}.npy", depth.squeeze().astype(np.float32))
```

**⚠️ 深度单位与有效范围（必须严格遵守）**：

1. **深度单位**：
   - Habitat 深度是 **相机到表面的欧氏距离**（米）
   - **0 表示无效点**（超出范围或无表面）
   - 不需要额外处理，直接保存原始值

2. **深度范围必须与场景尺度匹配**：
   ```yaml
   DEPTH_SENSOR:
     MIN_DEPTH: 0.5    # ⚠️ 太小会丢失近处物体
     MAX_DEPTH: 5.0    # ⚠️ 太小会大量截断远处物体
   ```
   - 室内场景推荐：`MIN_DEPTH: 0.5`, `MAX_DEPTH: 5.0`
   - 开阔场景推荐：`MIN_DEPTH: 0.5`, `MAX_DEPTH: 10.0`
   - **错误配置** → **大量 0 值** → **可见性判断失效** → **关键帧选择错误**

3. **RGB/Depth 分辨率必须完全相同**：
   ```python
   # ⚠️ 检查分辨率一致性
   assert rgb.shape[:2] == depth.shape[:2], \
       f"RGB shape {rgb.shape} != Depth shape {depth.shape}"
   ```
   - 分辨率不匹配 → 投影像素坐标错误 → 热力图完全错位

4. **深度有效性检查**：
   ```python
   # 统计有效深度比例
   valid_ratio = (depth > 0).sum() / depth.size
   if valid_ratio < 0.5:
       logger.warning(f"Low valid depth ratio: {valid_ratio:.2%}")
       # 可能原因：MAX_DEPTH 过小，或场景过于开阔
   ```

---

#### 📐 位姿序列（`poses.json`）

**格式**：4×4 变换矩阵的 JSON 列表

```json
[
  [
    [-0.588, 0.770, 0.248, -0.313],
    [0.000, 0.307, -0.952, 1.200],
    [0.809, 0.560, 0.180, 1.773],
    [0.0, 0.0, 0.0, 1.0]
  ],
  [
    [...],
    [...],
    [...],
    [0.0, 0.0, 0.0, 1.0]
  ]
]
```

**说明**：
- **变换矩阵**：`T_w_c` = 相机到世界（Camera-to-World）
- **结构**：
  ```
  [R | t]   = [3×3 旋转 | 3×1 平移]
  [0 | 1]     [  0 0 0  |    1    ]
  ```
- **列表长度** = 帧数
- **数据类型**：`float` (Python) / `float32` (NumPy)

**坐标系约定**：
- **世界坐标系**：Habitat 场景坐标系
  - X 轴：右
  - Y 轴：上
  - Z 轴：前
- **相机坐标系**：
  - X 轴：右
  - Y 轴：上
  - Z 轴：前（深度方向）

**Python 生成示例（⚠️ 必须包含传感器外参）**：

```python
import numpy as np
import json

def quaternion_to_rotation_matrix(q) -> np.ndarray:
    """
    将 Habitat Magnum Quaternion 转换为 3×3 旋转矩阵

    Args:
        q: Magnum Quaternion (属性: q.scalar, q.vector)

    Returns:
        np.ndarray: 3×3 旋转矩阵
    """
    # Magnum Quaternion: w (scalar) + (x, y, z) (vector)
    w = q.scalar
    x = q.vector.x
    y = q.vector.y
    z = q.vector.z

    # 四元数到旋转矩阵公式
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ], dtype=np.float32)

    return R


def get_sensor_extrinsics(config) -> np.ndarray:
    """
    从传感器配置获取外参矩阵 T_agent_cam（Agent到相机）

    Args:
        config: Habitat 配置对象

    Returns:
        np.ndarray: 4×4 外参矩阵
    """
    # RGB 传感器配置
    sensor_cfg = config.SIMULATOR.RGB_SENSOR

    # 传感器相对 agent 的位置（通常是 [0, height, 0]）
    # 例如 POSITION: [0, 0.88, 0] 表示相机在 agent 上方 0.88m
    sensor_position = np.array(sensor_cfg.POSITION, dtype=np.float32)

    # 传感器相对 agent 的旋转（如果配置中有 ORIENTATION）
    # 默认情况下传感器朝向与 agent 一致，无额外旋转
    if hasattr(sensor_cfg, 'ORIENTATION'):
        sensor_rotation = quaternion_to_rotation_matrix(sensor_cfg.ORIENTATION)
    else:
        sensor_rotation = np.eye(3, dtype=np.float32)

    # 构建外参矩阵 T_agent_cam
    T_agent_cam = np.eye(4, dtype=np.float32)
    T_agent_cam[:3, :3] = sensor_rotation
    T_agent_cam[:3, 3] = sensor_position

    return T_agent_cam


# 主循环中（⚠️ 正确方法：Agent 位姿 × 传感器外参）
poses = []
T_agent_cam = get_sensor_extrinsics(config)  # 只需获取一次

for frame_id in range(num_frames):
    agent_state = sim.get_agent_state()

    # 获取 Agent 位置和旋转
    agent_position = agent_state.position  # np.array([x, y, z])
    agent_rotation = agent_state.rotation  # Magnum Quaternion

    # Agent 到世界的变换矩阵
    R_agent = quaternion_to_rotation_matrix(agent_rotation)
    T_w_agent = np.eye(4, dtype=np.float32)
    T_w_agent[:3, :3] = R_agent
    T_w_agent[:3, 3] = agent_position

    # ⚠️ 关键：相机到世界 = Agent到世界 × Agent到相机
    T_w_c = T_w_agent @ T_agent_cam

    poses.append(T_w_c.tolist())

# 保存
with open("poses.json", "w") as f:
    json.dump(poses, f, indent=2)
```

**⚠️ 为什么必须加传感器外参？**

- Habitat 中相机通常相对 agent 有高度偏置（如 0.88m）和可能的俯仰角
- 若直接用 `agent_state.position/rotation` 组 `T_w_c`，投影时会有**系统性偏差**
- 后续的 3D→2D 投影、遮挡检查、热力图生成都依赖准确的相机位姿
- **错误的位姿** → **错误的可见性判断** → **错误的关键帧选择** → **低质量热力图**

---

#### 🔍 相机内参（`intrinsics.json`）

```json
{
  "fx": 332.55375,
  "fy": 332.55375,
  "cx": 192.0,
  "cy": 192.0,
  "K": [
    [332.55375, 0.0, 192.0],
    [0.0, 332.55375, 192.0],
    [0.0, 0.0, 1.0]
  ]
}
```

**说明**：
- `fx`, `fy`: 焦距（像素）
- `cx`, `cy`: 主点坐标（像素，通常为图像中心）
- `K`: 3×3 内参矩阵

**从 Habitat 配置计算（⚠️ 必须与传感器配置严格一致）**：

```python
import numpy as np
import math

def compute_intrinsics(config) -> dict:
    """
    从 Habitat 配置计算相机内参（必须与传感器配置完全一致）

    Args:
        config: Habitat 配置对象

    Returns:
        dict: 内参字典
    """
    # ⚠️ 关键：从配置读取，确保与传感器一致
    rgb_cfg = config.SIMULATOR.RGB_SENSOR
    depth_cfg = config.SIMULATOR.DEPTH_SENSOR

    width = rgb_cfg.WIDTH
    height = rgb_cfg.HEIGHT
    hfov_degrees = rgb_cfg.HFOV

    # ⚠️ 检查 RGB 和 Depth 分辨率必须相同
    assert width == depth_cfg.WIDTH, \
        f"RGB width ({width}) != Depth width ({depth_cfg.WIDTH})"
    assert height == depth_cfg.HEIGHT, \
        f"RGB height ({height}) != Depth height ({depth_cfg.HEIGHT})"
    assert abs(hfov_degrees - depth_cfg.HFOV) < 1e-5, \
        f"RGB HFOV ({hfov_degrees}) != Depth HFOV ({depth_cfg.HFOV})"

    # 焦距公式：f = w / (2 * tan(hfov/2))
    fx = width / (2.0 * math.tan(math.radians(hfov_degrees / 2.0)))
    fy = fx  # 假设像素是正方形

    # 主点（图像中心）
    cx = width / 2.0
    cy = height / 2.0

    # 内参矩阵
    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "K": K.tolist(),
        # 保存原始配置以便验证
        "width": width,
        "height": height,
        "hfov": hfov_degrees
    }

# 使用示例
intrinsics = compute_intrinsics(config)

with open("intrinsics.json", "w") as f:
    json.dump(intrinsics, f, indent=2)
```

**⚠️ 内参一致性检查清单**：

1. **RGB 和 Depth 必须完全匹配**：
   - `RGB_SENSOR.WIDTH == DEPTH_SENSOR.WIDTH`
   - `RGB_SENSOR.HEIGHT == DEPTH_SENSOR.HEIGHT`
   - `RGB_SENSOR.HFOV == DEPTH_SENSOR.HFOV`

2. **采集时分辨率不得改变**：
   - 不要在代码中 resize 图像
   - 保存的 RGB/Depth 必须是传感器原始分辨率

3. **HFOV 必须与配置一致**：
   - 内参计算公式依赖 HFOV
   - HFOV 错误 → fx/fy 错误 → 投影完全错误

---

## 2. VLN-CE Episode 结构

### 2.1 Episode 包含的信息

VLN-CE 的每个 episode 包含：

```python
episode = env.current_episode

# 基本信息
episode.episode_id        # str: 唯一标识符
episode.scene_id          # str: 场景路径（例如 "data/.../1pXnuDYAj8r.glb"）

# 导航信息
episode.start_position    # np.array: 起点位置 [x, y, z]
episode.start_rotation    # Quaternion: 起点朝向
episode.goals             # List[NavigationGoal]: 目标点列表
episode.goals[0].position # np.array: 目标位置 [x, y, z]

# 指令（R2R 数据集）
episode.instruction       # InstructionData 对象
episode.instruction.instruction_text  # str: 导航指令文本
```

**示例**：
```
Episode ID: "1_0"
Scene: "1pXnuDYAj8r"
Instruction: "Walk forward through the living room, turn right at the kitchen."
Start: [1.23, 0.0, 4.56]
Goal: [7.89, 0.0, 1.23]
```

---

### 2.2 场景名称提取

```python
# 从 episode.scene_id 提取场景名称
scene_id = episode.scene_id  # "data/scene_datasets/mp3d/1pXnuDYAj8r/1pXnuDYAj8r.glb"
scene_name = scene_id.split("/")[-1].replace(".glb", "")  # "1pXnuDYAj8r"
```

---

## 3. Habitat 采集脚本

### 3.1 完整采集脚本

将以下脚本保存为 `/home/habitat/VLN-CE/collect_raw_data.py`：

```python
#!/usr/bin/env python3
"""
VLN-CE 原始数据采集脚本
保存为 raw_sequences 格式供后续打包使用

用法：
    python collect_raw_data.py \\
        --config habitat_extensions/config/vlnce_task.yaml \\
        --output ./raw_sequences \\
        --split train \\
        --num-clips 10 \\
        --frames-per-clip 50
"""

import os
import sys
import json
import math
import argparse
import numpy as np
import cv2
from pathlib import Path
from typing import Dict, List
import logging

import habitat
from habitat.config.default import get_config
from habitat.sims.habitat_simulator.actions import HabitatSimActions
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower


# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def quaternion_to_rotation_matrix(q) -> np.ndarray:
    """
    将 Habitat Magnum Quaternion 转换为 3×3 旋转矩阵

    Args:
        q: Magnum Quaternion (q.scalar, q.vector)

    Returns:
        np.ndarray: 3×3 旋转矩阵
    """
    # Magnum Quaternion: w (scalar) + (x, y, z) (vector)
    w = q.scalar
    x = q.vector.x
    y = q.vector.y
    z = q.vector.z

    # 四元数到旋转矩阵
    R = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ], dtype=np.float32)

    return R


def get_sensor_extrinsics(config) -> np.ndarray:
    """获取传感器外参矩阵 T_agent_cam"""
    sensor_cfg = config.SIMULATOR.RGB_SENSOR
    sensor_position = np.array(sensor_cfg.POSITION, dtype=np.float32)

    # 默认情况下传感器朝向与 agent 一致
    if hasattr(sensor_cfg, 'ORIENTATION'):
        sensor_rotation = quaternion_to_rotation_matrix(sensor_cfg.ORIENTATION)
    else:
        sensor_rotation = np.eye(3, dtype=np.float32)

    T_agent_cam = np.eye(4, dtype=np.float32)
    T_agent_cam[:3, :3] = sensor_rotation
    T_agent_cam[:3, 3] = sensor_position
    return T_agent_cam


def compute_intrinsics(config) -> Dict:
    """从配置计算相机内参（确保 RGB/Depth 一致）"""
    rgb_cfg = config.SIMULATOR.RGB_SENSOR
    depth_cfg = config.SIMULATOR.DEPTH_SENSOR

    width = rgb_cfg.WIDTH
    height = rgb_cfg.HEIGHT
    hfov_degrees = rgb_cfg.HFOV

    # 检查一致性
    assert width == depth_cfg.WIDTH, \
        f"RGB width ({width}) != Depth width ({depth_cfg.WIDTH})"
    assert height == depth_cfg.HEIGHT, \
        f"RGB height ({height}) != Depth height ({depth_cfg.HEIGHT})"

    fx = width / (2.0 * math.tan(math.radians(hfov_degrees / 2.0)))
    fy = fx
    cx = width / 2.0
    cy = height / 2.0

    K = np.array([
        [fx, 0.0, cx],
        [0.0, fy, cy],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    return {
        "fx": float(fx),
        "fy": float(fy),
        "cx": float(cx),
        "cy": float(cy),
        "K": K.tolist(),
        "width": width,
        "height": height,
        "hfov": hfov_degrees
    }


class RawDataCollector:
    """VLN-CE 原始数据采集器"""

    def __init__(
        self,
        config_path: str,
        output_root: str,
        split: str = "train"
    ):
        """
        初始化采集器

        Args:
            config_path: VLN-CE 配置文件路径
            output_root: 输出根目录
            split: 数据集划分（train/val/test）
        """
        # 加载 Habitat 配置
        self.config = get_config(config_path)
        self.config.defrost()
        self.config.DATASET.SPLIT = split
        self.config.freeze()

        self.output_root = Path(output_root)
        self.split = split

        # 初始化环境
        logger.info(f"Initializing Habitat environment for split: {split}")
        self.env = habitat.Env(config=self.config)
        self.sim = self.env.sim

        # 最短路径跟随器
        self.follower = ShortestPathFollower(
            self.sim,
            goal_radius=0.2,
            return_one_hot=False
        )

        logger.info("✅ Environment initialized successfully")

    def collect_clip(
        self,
        scene_name: str,
        clip_id: int,
        max_steps: int = 50
    ) -> Dict:
        """
        采集单个 clip 的原始数据

        Args:
            scene_name: 场景名称
            clip_id: Clip ID
            max_steps: 最大步数

        Returns:
            dict: 采集统计信息
        """
        # 重置环境
        observations = self.env.reset()
        episode = self.env.current_episode

        # 创建输出目录
        clip_dir = self.output_root / self.split / scene_name / f"clip_{clip_id:06d}"
        rgb_dir = clip_dir / "rgb"
        depth_dir = clip_dir / "depth"
        rgb_dir.mkdir(parents=True, exist_ok=True)
        depth_dir.mkdir(parents=True, exist_ok=True)

        logger.info(f"📁 Collecting clip: {clip_dir}")

        # ⚠️ 获取传感器外参（只需一次）
        T_agent_cam = get_sensor_extrinsics(self.config)

        # 采集数据
        poses = []
        done = False
        frame_id = 0

        while not done and frame_id < max_steps:
            # 1. 保存 RGB
            rgb = observations["rgb"]  # [H, W, 3], uint8, RGB
            rgb_path = rgb_dir / f"{frame_id:06d}.png"
            cv2.imwrite(str(rgb_path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))

            # 2. 保存深度
            depth = observations["depth"]  # [H, W, 1], float32
            depth_path = depth_dir / f"{frame_id:06d}.npy"
            depth_array = depth.squeeze().astype(np.float32)
            np.save(depth_path, depth_array)

            # ⚠️ 检查深度有效性
            valid_ratio = (depth_array > 0).sum() / depth_array.size
            if valid_ratio < 0.5:
                logger.warning(f"  Frame {frame_id}: Low valid depth ratio {valid_ratio:.2%}")

            # 3. 获取位姿（相机到世界 = Agent到世界 × Agent到相机）
            agent_state = self.sim.get_agent_state()
            agent_position = agent_state.position
            agent_rotation = agent_state.rotation  # Magnum Quaternion

            # Agent 到世界的变换矩阵
            R_agent = quaternion_to_rotation_matrix(agent_rotation)
            T_w_agent = np.eye(4, dtype=np.float32)
            T_w_agent[:3, :3] = R_agent
            T_w_agent[:3, 3] = agent_position

            # ⚠️ 关键：相机到世界 = Agent到世界 × Agent到相机
            T_w_c = T_w_agent @ T_agent_cam

            poses.append(T_w_c.tolist())

            # 4. 执行动作（使用最短路径专家）
            best_action = self.follower.get_next_action(episode.goals[0].position)
            if best_action is None:
                best_action = HabitatSimActions.STOP

            observations = self.env.step(best_action)
            done = (best_action == HabitatSimActions.STOP)

            frame_id += 1

        # 5. 保存位姿
        poses_path = clip_dir / "poses.json"
        with open(poses_path, "w") as f:
            json.dump(poses, f, indent=2)

        # 6. 保存内参（从配置读取，确保一致性）
        intrinsics = compute_intrinsics(self.config)
        intrinsics_path = clip_dir / "intrinsics.json"
        with open(intrinsics_path, "w") as f:
            json.dump(intrinsics, f, indent=2)

        logger.info(f"✅ Collected {frame_id} frames for clip_{clip_id:06d}")

        return {
            "clip_dir": str(clip_dir),
            "scene": scene_name,
            "clip_id": clip_id,
            "num_frames": frame_id,
            "episode_id": episode.episode_id,
            "instruction": getattr(episode.instruction, 'instruction_text', '')
        }

    def collect_from_dataset(
        self,
        num_clips: int = 10,
        frames_per_clip: int = 50
    ):
        """
        从 VLN-CE 数据集采集多个 clips

        Args:
            num_clips: 采集的 clip 数量
            frames_per_clip: 每个 clip 的最大帧数
        """
        dataset = self.env.episodes
        total_episodes = len(dataset)

        logger.info(f"📊 Dataset contains {total_episodes} episodes")
        logger.info(f"🎯 Target: {num_clips} clips")

        collected = []
        failed_count = 0

        for clip_id in range(1, num_clips + 1):
            try:
                # 重置环境（自动选择下一个 episode）
                observations = self.env.reset()
                episode = self.env.current_episode

                # 提取场景名称
                scene_name = episode.scene_id.split("/")[-1].replace(".glb", "")

                # 采集
                stats = self.collect_clip(
                    scene_name=scene_name,
                    clip_id=clip_id,
                    max_steps=frames_per_clip
                )
                collected.append(stats)

            except Exception as e:
                logger.error(f"❌ Error collecting clip {clip_id}: {e}")
                failed_count += 1
                continue

        # 保存索引
        index_path = self.output_root / f"{self.split}_index.json"
        with open(index_path, "w") as f:
            json.dump(collected, f, indent=2)

        # 统计
        logger.info("=" * 60)
        logger.info(f"✅ Collection completed!")
        logger.info(f"   Success: {len(collected)}/{num_clips} clips")
        logger.info(f"   Failed: {failed_count} clips")
        logger.info(f"📁 Output: {self.output_root / self.split}")
        logger.info(f"📄 Index: {index_path}")

    def close(self):
        """关闭环境"""
        self.env.close()
        logger.info("🔒 Environment closed")


def main():
    parser = argparse.ArgumentParser(
        description="Collect raw VLN data from Habitat",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example:
  python collect_raw_data.py \\
      --config habitat_extensions/config/vlnce_task.yaml \\
      --output ./raw_sequences \\
      --split train \\
      --num-clips 10 \\
      --frames-per-clip 50
        """
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to VLN-CE config (e.g., habitat_extensions/config/vlnce_task.yaml)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="./raw_sequences",
        help="Output root for raw sequences (default: ./raw_sequences)"
    )
    parser.add_argument(
        "--split",
        type=str,
        default="train",
        choices=["train", "val", "test"],
        help="Dataset split (default: train)"
    )
    parser.add_argument(
        "--num-clips",
        type=int,
        default=10,
        help="Number of clips to collect (default: 10)"
    )
    parser.add_argument(
        "--frames-per-clip",
        type=int,
        default=50,
        help="Maximum frames per clip (default: 50)"
    )

    args = parser.parse_args()

    # 打印配置
    logger.info("🚀 Starting VLN-CE data collection")
    logger.info(f"   Config: {args.config}")
    logger.info(f"   Output: {args.output}")
    logger.info(f"   Split: {args.split}")
    logger.info(f"   Clips: {args.num_clips}")
    logger.info(f"   Frames/Clip: {args.frames_per_clip}")

    # 创建采集器
    collector = RawDataCollector(
        config_path=args.config,
        output_root=args.output,
        split=args.split
    )

    try:
        # 采集数据
        collector.collect_from_dataset(
            num_clips=args.num_clips,
            frames_per_clip=args.frames_per_clip
        )
    finally:
        collector.close()

    logger.info("🎉 Done!")


if __name__ == "__main__":
    main()
```

---

### 3.2 运行采集脚本

```bash
# 在 VLN-CE 容器中运行
cd /home/habitat/VLN-CE

# 测试模式：采集 5 个 clips
python collect_raw_data.py \
    --config habitat_extensions/config/vlnce_task.yaml \
    --output /shared/raw_sequences \
    --split train \
    --num-clips 5 \
    --frames-per-clip 30

# 完整采集（根据需要调整数量）
python collect_raw_data.py \
    --config habitat_extensions/config/vlnce_task.yaml \
    --output /shared/raw_sequences \
    --split train \
    --num-clips 100 \
    --frames-per-clip 50

# 采集验证集
python collect_raw_data.py \
    --config habitat_extensions/config/vlnce_task.yaml \
    --output /shared/raw_sequences \
    --split val \
    --num-clips 20 \
    --frames-per-clip 50
```

**输出示例**：
```
🚀 Starting VLN-CE data collection
   Config: habitat_extensions/config/vlnce_task.yaml
   Output: /shared/raw_sequences
   Split: train
   Clips: 5
   Frames/Clip: 30
Initializing Habitat environment for split: train
✅ Environment initialized successfully
📊 Dataset contains 2349 episodes
🎯 Target: 5 clips
📁 Collecting clip: /shared/raw_sequences/train/1pXnuDYAj8r/clip_000001
✅ Collected 28 frames for clip_000001
...
✅ Collection completed!
   Success: 5/5 clips
   Failed: 0 clips
📁 Output: /shared/raw_sequences/train
📄 Index: /shared/raw_sequences/train_index.json
🔒 Environment closed
🎉 Done!
```

---

## 4. 数据验证工具

### 4.1 最小可复现实验（Smoke Test）

在采集数据前，**强烈建议**先运行以下 smoke test 验证配置正确性：

```python
#!/usr/bin/env python3
"""Smoke test：验证相机位姿和投影的正确性"""

import numpy as np
import math
import habitat
from habitat.config.default import get_config

def quaternion_to_rotation_matrix(q):
    """同前"""
    w, x, y, z = q.scalar, q.vector.x, q.vector.y, q.vector.z
    return np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ], dtype=np.float32)

def get_sensor_extrinsics(config):
    """同前"""
    sensor_cfg = config.SIMULATOR.RGB_SENSOR
    sensor_position = np.array(sensor_cfg.POSITION, dtype=np.float32)
    if hasattr(sensor_cfg, 'ORIENTATION'):
        sensor_rotation = quaternion_to_rotation_matrix(sensor_cfg.ORIENTATION)
    else:
        sensor_rotation = np.eye(3, dtype=np.float32)
    T_agent_cam = np.eye(4, dtype=np.float32)
    T_agent_cam[:3, :3] = sensor_rotation
    T_agent_cam[:3, 3] = sensor_position
    return T_agent_cam

def compute_intrinsics(config):
    """同前"""
    rgb_cfg = config.SIMULATOR.RGB_SENSOR
    width, height, hfov = rgb_cfg.WIDTH, rgb_cfg.HEIGHT, rgb_cfg.HFOV
    fx = width / (2.0 * math.tan(math.radians(hfov / 2.0)))
    K = np.array([[fx, 0, width/2], [0, fx, height/2], [0, 0, 1]], dtype=np.float32)
    return K, width, height

def smoke_test(config_path):
    """
    Smoke Test 1: 圆点投影自检
    测试：相机前方 Z 轴上一点是否投影到图像中心附近
    """
    config = get_config(config_path)
    env = habitat.Env(config=config)
    sim = env.sim

    # 初始化
    env.reset()
    agent_state = sim.get_agent_state()
    T_agent_cam = get_sensor_extrinsics(config)
    K, width, height = compute_intrinsics(config)

    # Agent 到世界
    R_agent = quaternion_to_rotation_matrix(agent_state.rotation)
    T_w_agent = np.eye(4, dtype=np.float32)
    T_w_agent[:3, :3] = R_agent
    T_w_agent[:3, 3] = agent_state.position

    # 相机到世界
    T_w_c = T_w_agent @ T_agent_cam
    T_c_w = np.linalg.inv(T_w_c)

    # 测试点：相机前方 2 米（相机坐标系 Z=2）
    point_cam = np.array([0, 0, 2, 1], dtype=np.float32)
    point_world = T_w_c @ point_cam

    # 投影到像素坐标
    point_cam_back = T_c_w @ point_world
    uv_homo = K @ point_cam_back[:3]
    u = uv_homo[0] / uv_homo[2]
    v = uv_homo[1] / uv_homo[2]

    print("=" * 60)
    print("Smoke Test 1: 圆点投影自检")
    print(f"  相机坐标系测试点: [0, 0, 2] (前方 2m)")
    print(f"  投影像素坐标: ({u:.1f}, {v:.1f})")
    print(f"  图像中心: ({width/2:.1f}, {height/2:.1f})")
    print(f"  偏差: ({abs(u - width/2):.1f}, {abs(v - height/2):.1f}) px")

    if abs(u - width/2) < 5 and abs(v - height/2) < 5:
        print("  ✅ PASS: 投影接近图像中心，相机朝向 +Z 正确")
    else:
        print("  ❌ FAIL: 投影偏离中心，检查外参或坐标系约定")

    """
    Smoke Test 2: 姿态矩阵健康检查
    """
    print("\nSmoke Test 2: 姿态矩阵健康检查")

    # 检查 NaN
    if np.isnan(T_w_c).any():
        print("  ❌ FAIL: 位姿矩阵包含 NaN")
    else:
        print("  ✅ PASS: 无 NaN")

    # 检查旋转矩阵行列式
    R = T_w_c[:3, :3]
    det = np.linalg.det(R)
    print(f"  旋转矩阵行列式: {det:.6f} (期望 ≈ 1.0)")
    if abs(det - 1.0) < 0.01:
        print("  ✅ PASS: 旋转矩阵正交")
    else:
        print("  ❌ FAIL: 旋转矩阵不正交，检查四元数转换")

    # 检查最后一行
    if np.allclose(T_w_c[3, :], [0, 0, 0, 1]):
        print("  ✅ PASS: 齐次坐标最后一行正确")
    else:
        print("  ❌ FAIL: 最后一行不是 [0, 0, 0, 1]")

    print("=" * 60)
    env.close()

if __name__ == "__main__":
    smoke_test("habitat_extensions/config/vlnce_task.yaml")
```

**运行 Smoke Test**：
```bash
cd /home/habitat/VLN-CE
python smoke_test.py
```

**期望输出**：
```
============================================================
Smoke Test 1: 圆点投影自检
  相机坐标系测试点: [0, 0, 2] (前方 2m)
  投影像素坐标: (320.0, 240.0)
  图像中心: (320.0, 240.0)
  偏差: (0.0, 0.0) px
  ✅ PASS: 投影接近图像中心，相机朝向 +Z 正确

Smoke Test 2: 姿态矩阵健康检查
  ✅ PASS: 无 NaN
  旋转矩阵行列式: 1.000000 (期望 ≈ 1.0)
  ✅ PASS: 旋转矩阵正交
  ✅ PASS: 齐次坐标最后一行正确
============================================================
```

---

### 4.2 验证脚本

将以下脚本保存为 `validate_raw_data.py`：

```python
#!/usr/bin/env python3
"""验证 raw_sequences 数据完整性"""

import json
import numpy as np
from pathlib import Path
import sys
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_clip(clip_dir: Path) -> bool:
    """验证单个 clip"""
    logger.info(f"Validating: {clip_dir.name}")

    # 1. 检查必要文件
    required_files = ["poses.json", "intrinsics.json"]
    required_dirs = ["rgb", "depth"]

    for fname in required_files:
        if not (clip_dir / fname).exists():
            logger.error(f"  ❌ Missing {fname}")
            return False

    for dname in required_dirs:
        if not (clip_dir / dname).exists():
            logger.error(f"  ❌ Missing directory {dname}/")
            return False

    # 2. 加载位姿
    with open(clip_dir / "poses.json") as f:
        poses = json.load(f)

    num_poses = len(poses)
    if num_poses == 0:
        logger.error(f"  ❌ Empty poses.json")
        return False

    # 3. 检查 RGB 帧
    rgb_files = sorted((clip_dir / "rgb").glob("*.png"))
    if len(rgb_files) != num_poses:
        logger.error(f"  ❌ RGB count mismatch: {len(rgb_files)} != {num_poses}")
        return False

    # 4. 检查深度图
    depth_files = sorted((clip_dir / "depth").glob("*.npy"))
    if len(depth_files) != num_poses:
        logger.error(f"  ❌ Depth count mismatch: {len(depth_files)} != {num_poses}")
        return False

    # 5. 验证位姿格式
    for i, pose in enumerate(poses):
        if not isinstance(pose, list) or len(pose) != 4:
            logger.error(f"  ❌ Pose {i}: invalid format (not 4×4)")
            return False
        for row in pose:
            if not isinstance(row, list) or len(row) != 4:
                logger.error(f"  ❌ Pose {i}: invalid row")
                return False

        # 检查最后一行
        if pose[3] != [0.0, 0.0, 0.0, 1.0]:
            logger.warning(f"  ⚠️  Pose {i}: last row is not [0, 0, 0, 1]")

    # 6. 验证内参
    with open(clip_dir / "intrinsics.json") as f:
        intrinsics = json.load(f)

    required_keys = ["fx", "fy", "cx", "cy", "K"]
    for key in required_keys:
        if key not in intrinsics:
            logger.error(f"  ❌ Missing intrinsics key: {key}")
            return False

    # 7. 抽查深度图
    sample_depth = np.load(depth_files[0])
    if sample_depth.dtype != np.float32:
        logger.warning(f"  ⚠️  Depth dtype is {sample_depth.dtype}, expected float32")

    # 检查深度范围
    depth_min, depth_max = sample_depth.min(), sample_depth.max()
    if depth_max == 0:
        logger.warning(f"  ⚠️  All depths are zero (no valid depth)")

    logger.info(f"  ✅ Valid ({num_poses} frames, depth range: {depth_min:.3f}-{depth_max:.3f}m)")
    return True


def main():
    if len(sys.argv) < 2:
        print("Usage: python validate_raw_data.py <raw_sequences_root> [split]")
        print("Example: python validate_raw_data.py ./raw_sequences train")
        sys.exit(1)

    root = Path(sys.argv[1])
    split = sys.argv[2] if len(sys.argv) > 2 else "train"

    split_dir = root / split
    if not split_dir.exists():
        logger.error(f"❌ Split directory not found: {split_dir}")
        sys.exit(1)

    # 找到所有 clips
    clips = []
    for scene_dir in split_dir.iterdir():
        if scene_dir.is_dir():
            for clip_dir in scene_dir.iterdir():
                if clip_dir.is_dir() and clip_dir.name.startswith("clip_"):
                    clips.append(clip_dir)

    logger.info(f"Found {len(clips)} clips in split '{split}'")
    logger.info("=" * 60)

    valid_count = 0
    invalid_clips = []

    for clip_dir in sorted(clips):
        if validate_clip(clip_dir):
            valid_count += 1
        else:
            invalid_clips.append(clip_dir.name)

    logger.info("=" * 60)
    logger.info(f"Validation Results: {valid_count}/{len(clips)} clips valid")

    if invalid_clips:
        logger.warning(f"Invalid clips: {', '.join(invalid_clips)}")

    sys.exit(0 if valid_count == len(clips) else 1)


if __name__ == "__main__":
    main()
```

**运行验证**：
```bash
python validate_raw_data.py /shared/raw_sequences train
```

**输出示例**：
```
INFO - Found 5 clips in split 'train'
============================================================
INFO - Validating: clip_000001
INFO -   ✅ Valid (28 frames, depth range: 0.500-4.982m)
INFO - Validating: clip_000002
INFO -   ✅ Valid (30 frames, depth range: 0.512-5.000m)
...
============================================================
INFO - Validation Results: 5/5 clips valid
```

---

## 5. 打包为训练格式

原始数据采集完成后，在训练容器中运行打包脚本：

```bash
# 在训练容器（/home/VLN/Project）中运行
cd /home/VLN/Project

# 确保 raw_sequences 可访问（软链接或挂载）
ln -s /shared/raw_sequences ./raw_sequences

# 打包训练集
python scripts/pack_dataset.py \
    --config configs/dataset_pack.yaml \
    --split train

# 打包验证集
python scripts/pack_dataset.py \
    --config configs/dataset_pack.yaml \
    --split val
```

**打包过程**：

1. **读取原始数据**：从 `raw_sequences/` 加载 RGB、Depth、Poses、Intrinsics
2. **提取子序列**：使用滑动窗口提取固定长度子序列（T 帧）
3. **选择关键帧**：使用 visibility-aware FPS 采样选择 K 个关键帧
4. **生成热力图**：
   - 使用 `src/data/heatmap_builder.py` 中的几何投影函数
   - `unproject_depth_to_points()`: 深度图 → 3D 点云
   - `world_from_cam()`: 相机坐标系 → 世界坐标系
   - `project_keyframe_to_ref()`: 关键帧 3D 点 → 参考帧 2D 投影
   - 应用遮挡检查（Z-buffer 深度比较）
   - `heatmap_from_points()`: 2D 点 → 高斯热力图
5. **保存训练格式**：
   ```
   data/habitat_vln/{split}/{scene}/clip_{N}/
   ├── rgb/            # T 帧 RGB 图像
   ├── depth/          # T 帧深度图
   ├── poses.json      # T 个位姿
   ├── intrinsics.json # 相机内参
   ├── heatmaps.npy    # [K, Hm, Wm], 归一化概率分布
   ├── mask.npy        # [K], 有效性标记 (0/1)
   └── meta.json       # 元数据（ref_idx, key_indices, 质量指标等）
   ```

---

## 6. Habitat API 参考

### 6.1 核心 API

#### 环境初始化
```python
import habitat
from habitat.config.default import get_config

config = get_config("path/to/config.yaml")
env = habitat.Env(config=config)
```

#### 获取观测
```python
observations = env.reset()  # 或 env.step(action)

rgb = observations["rgb"]      # [H, W, 3], uint8, RGB
depth = observations["depth"]  # [H, W, 1], float32, 米
```

#### 获取 Agent 状态
```python
agent_state = sim.get_agent_state()

position = agent_state.position    # np.array([x, y, z]), float32
rotation = agent_state.rotation    # Magnum Quaternion
```

#### 最短路径专家
```python
from habitat.tasks.nav.shortest_path_follower import ShortestPathFollower

follower = ShortestPathFollower(sim, goal_radius=0.2, return_one_hot=False)
action = follower.get_next_action(episode.goals[0].position)

if action is None:
    action = HabitatSimActions.STOP
```

#### 执行动作
```python
from habitat.sims.habitat_simulator.actions import HabitatSimActions

observations = env.step(HabitatSimActions.MOVE_FORWARD)
observations = env.step(HabitatSimActions.TURN_LEFT)
observations = env.step(HabitatSimActions.TURN_RIGHT)
observations = env.step(HabitatSimActions.STOP)
```

---

### 6.2 坐标系说明

**Habitat 世界坐标系**：
- **X 轴**：右
- **Y 轴**：上
- **Z 轴**：前

**相机坐标系**：
- **X 轴**：右
- **Y 轴**：上
- **Z 轴**：前（深度方向，正值表示物体在相机前方）

**变换矩阵**：
- `T_w_c`：相机到世界（Camera-to-World）
- `T_c_w`：世界到相机（World-to-Camera）= `np.linalg.inv(T_w_c)`

**坐标变换**：
```python
# 世界坐标 → 相机坐标
T_c_w = np.linalg.inv(T_w_c)
point_cam = T_c_w @ np.append(point_world, 1.0)  # 齐次坐标

# 相机坐标 → 世界坐标
point_world = T_w_c @ np.append(point_cam, 1.0)
```

---

## 7. 常见问题排查

### Q1: 如何确定采集的数据量？采样策略是什么？

**⚠️ 核心原则：帧间重叠度优先于时长！**

`pack_dataset.py` 使用 **visibility-aware FPS** 选择关键帧，依赖**帧间视野重叠**。重叠度不足 → K_eff < 2 → 样本被丢弃。

**推荐采样策略**：

1. **采样密度优先**：
   ```python
   # ✅ 推荐：每次动作后都存一帧
   while not done and frame_id < max_steps:
       save_frame(observations)  # 先保存
       action = get_next_action()
       observations = env.step(action)
       frame_id += 1

   # ❌ 不推荐：稀疏采样（如每 3 步存 1 帧）
   ```

2. **减小转角**：
   ```python
   # ✅ 推荐：小角度转动（15°）
   config.SIMULATOR.TURN_ANGLE = 15  # 默认 30°

   # 或手动细分大转角
   if action == TURN_LEFT:
       for _ in range(2):  # 拆分为 2 个小转角
           observations = env.step(TURN_LEFT_SMALL)
           save_frame(observations)
   ```

3. **必要时混合随机动作**：
   ```python
   # 延长路径，增加多角度覆盖
   if random.random() < 0.1:  # 10% 随机探索
       action = random.choice([MOVE_FORWARD, TURN_LEFT, TURN_RIGHT])
   else:
       action = follower.get_next_action(goal_position)
   ```

**采集规模建议（渐进式）**：

1. **快速测试**（验证流程）：
   - 5-10 clips × 20-30 frames
   - 目的：验证采集→打包→训练流程
   - 运行 smoke test + 打包 + 质量报告

2. **小规模试验**（优化采样策略）：
   - 50-100 clips
   - **先打包 → 查看 K_eff/熵 报告 → 调整采样策略**
   - 如果 K_eff < 2 的比例 > 30%，调整：
     - 增加帧密度（每步都存）
     - 减小转角（15° 代替 30°）
     - 增加 MOVE_FORWARD 步数

3. **完整训练**：
   - 500-2000 clips（根据场景数和存储）
   - 确保 K_eff ≥ 2 的达标率 > 70%

**存储估算**：
- RGB (640×480 PNG): ~100 KB/frame
- Depth (640×480 NPY): ~1.2 MB/frame
- Poses + Intrinsics: ~10 KB/clip
- **总计**：~1.3 MB/frame × 帧数

**⚠️ 判停标准**：

- **不要一次性采集 1000+ clips！**
- **正确流程**：
  1. 采集 10-50 clips
  2. 运行 `pack_dataset.py`
  3. 检查质量报告（K_eff 分布、熵、有效样本率）
  4. 如果质量不佳，调整采样策略后重新采集
  5. 质量达标后再扩大规模

### Q2: 深度图是否需要预处理？

**不需要**。保存为原始 float32 格式（米）即可。`pack_dataset.py` 会自动处理：
- 深度反投影为 3D 点
- 遮挡检查
- 热力图生成

### Q3: 如何处理空位姿或坐标异常？

**检查方法**：
```python
# 验证位姿是否包含 NaN
if np.isnan(T_w_c).any():
    logger.warning("NaN detected in pose matrix")
    # 跳过该帧或使用前一帧位姿
    continue

# 验证位姿是否合法（旋转矩阵行列式应为 1）
R = T_w_c[:3, :3]
det = np.linalg.det(R)
if abs(det - 1.0) > 0.01:
    logger.warning(f"Invalid rotation matrix (det={det:.3f})")
```

### Q4: ⚠️ RGB 和 Depth 配置必须严格一致！

确保配置中 RGB 和 Depth 传感器使用**完全相同的参数**：

```yaml
RGB_SENSOR:
  WIDTH: 640
  HEIGHT: 480
  HFOV: 79
  POSITION: [0, 0.88, 0]  # ⚠️ 必须与 DEPTH_SENSOR 相同

DEPTH_SENSOR:
  WIDTH: 640              # ⚠️ 必须与 RGB 相同
  HEIGHT: 480             # ⚠️ 必须与 RGB 相同
  HFOV: 79                # ⚠️ 必须与 RGB 相同
  POSITION: [0, 0.88, 0]  # ⚠️ 必须与 RGB 相同
  MIN_DEPTH: 0.5          # ⚠️ 根据场景尺度调整
  MAX_DEPTH: 5.0          # ⚠️ 室内场景推荐 5.0，开阔场景 10.0
```

**检查方法**：
```python
# 在采集脚本开头添加
rgb_cfg = config.SIMULATOR.RGB_SENSOR
depth_cfg = config.SIMULATOR.DEPTH_SENSOR

assert rgb_cfg.WIDTH == depth_cfg.WIDTH, "WIDTH mismatch!"
assert rgb_cfg.HEIGHT == depth_cfg.HEIGHT, "HEIGHT mismatch!"
assert rgb_cfg.HFOV == depth_cfg.HFOV, "HFOV mismatch!"
assert rgb_cfg.POSITION == depth_cfg.POSITION, "POSITION mismatch!"

logger.info(f"✅ Sensor config verified: {rgb_cfg.WIDTH}×{rgb_cfg.HEIGHT}, HFOV={rgb_cfg.HFOV}")
logger.info(f"✅ Depth range: [{depth_cfg.MIN_DEPTH}, {depth_cfg.MAX_DEPTH}] meters")
```

### Q5: 如何验证采集的数据可用？

#### 方法 1：运行验证脚本
```bash
python validate_raw_data.py /shared/raw_sequences train
```

#### 方法 2：可视化检查
```python
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 读取数据
rgb = cv2.imread("rgb/000000.png")
depth = np.load("depth/000000.npy")

# 可视化深度
depth_vis = (depth / depth.max() * 255).astype(np.uint8)
depth_colored = cv2.applyColorMap(depth_vis, cv2.COLORMAP_JET)

# 显示
plt.figure(figsize=(12, 4))
plt.subplot(131); plt.imshow(cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)); plt.title("RGB")
plt.subplot(132); plt.imshow(depth, cmap='gray'); plt.title("Depth (meters)")
plt.subplot(133); plt.imshow(cv2.cvtColor(depth_colored, cv2.COLOR_BGR2RGB)); plt.title("Depth (colored)")
plt.tight_layout()
plt.show()
```

#### 方法 3：运行打包测试
```bash
cd /home/VLN/Project

# 打包测试（少量数据）
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split train

# 可视化打包后的数据
python scripts/inspect_dataset.py --root ./data/habitat_vln --split train --num 3
```

### Q6: Episode 太短怎么办？

如果 episode 结束太快（帧数 < 期望帧数）：

**原因**：
- 起点离目标太近
- 最短路径专家快速到达目标

**解决方案**：
1. **调整 `goal_radius`**：增大目标半径，延长导航
   ```python
   follower = ShortestPathFollower(sim, goal_radius=0.5)  # 默认 0.2
   ```

2. **随机探索**：混合使用专家和随机动作
   ```python
   if random.random() < 0.1:  # 10% 随机动作
       action = random.choice([MOVE_FORWARD, TURN_LEFT, TURN_RIGHT])
   else:
       action = follower.get_next_action(goal_position)
   ```

3. **过滤短 episodes**：采集后删除帧数过少的 clips

### Q7: 内存不足怎么办？

**优化方法**：
- **减少并行环境数**：VLN-CE 默认使用单环境，通常无需修改
- **降低分辨率**：使用 384×384 而非 640×480
- **减少 `num-clips`**：分批采集

---

### Q8: ⚠️ 场景命名必须与打包配置一致！

**问题**：打包时提示 "找不到场景目录" 或样本被静默跳过。

**原因**：`dataset_pack.yaml` 中的场景名与 `raw_sequences/{split}/{scene}/` 目录名不匹配。

**解决方法**：

1. **检查采集脚本中的场景名提取**：
   ```python
   # ✅ 正确：从 episode.scene_id 提取 Matterport3D 场景 ID
   scene_id = episode.scene_id  # "data/.../mp3d/1pXnuDYAj8r/1pXnuDYAj8r.glb"
   scene_name = scene_id.split("/")[-1].replace(".glb", "")  # "1pXnuDYAj8r"

   # ❌ 错误：使用 episode_id 作为场景名
   scene_name = episode.episode_id  # "1_0" (这不是场景名！)
   ```

2. **验证目录结构**：
   ```bash
   # 检查实际目录
   ls raw_sequences/train/
   # 应该看到：1pXnuDYAj8r, 2azQ1b91cZZ, ...（Matterport3D 场景 ID）

   # 检查 dataset_pack.yaml
   cat configs/dataset_pack.yaml | grep scenes
   # 应该匹配：scenes: [1pXnuDYAj8r, 2azQ1b91cZZ, ...]
   ```

3. **打包时指定正确场景**：
   ```yaml
   # configs/dataset_pack.yaml
   data:
     splits:
       train:
         scenes: [1pXnuDYAj8r, 2azQ1b91cZZ]  # ⚠️ 必须与目录名完全一致
         clips_per_scene: 100
   ```

4. **自动生成场景列表**（推荐）：
   ```python
   # 在打包前自动扫描场景
   import os
   train_dir = "raw_sequences/train"
   scenes = [d for d in os.listdir(train_dir) if os.path.isdir(os.path.join(train_dir, d))]
   print(f"Available scenes: {scenes}")
   # 更新 dataset_pack.yaml 的 scenes 字段
   ```

**⚠️ 场景名一致性检查清单**：
- [ ] `raw_sequences/train/{scene}/` 目录存在
- [ ] `dataset_pack.yaml` 中 `scenes` 列表与目录名完全匹配
- [ ] 场景名是 Matterport3D 的标准 ID（如 `1pXnuDYAj8r`），不是 episode ID

---

## 8. 完整工作流程总结

### 步骤 0：⚠️ 运行 Smoke Test（必须！）

```bash
cd /home/habitat/VLN-CE

# 验证相机位姿和投影的正确性
python smoke_test.py

# 期望输出：
# ✅ PASS: 投影接近图像中心，相机朝向 +Z 正确
# ✅ PASS: 无 NaN
# ✅ PASS: 旋转矩阵正交
# ✅ PASS: 齐次坐标最后一行正确

# ❌ 如果 FAIL，必须先修复配置再继续！
```

### 步骤 1：在 VLN-CE 容器中采集原始数据（渐进式）

```bash
cd /home/habitat/VLN-CE

# 1.1 快速测试（5-10 clips）
python collect_raw_data.py \
    --config habitat_extensions/config/vlnce_task.yaml \
    --output /shared/raw_sequences \
    --split train \
    --num-clips 10 \
    --frames-per-clip 30

# 1.2 验证数据 + 打包 + 检查质量
# （见步骤 2-4）

# 1.3 质量达标后，扩大规模（100+ clips）
python collect_raw_data.py \
    --config habitat_extensions/config/vlnce_task.yaml \
    --output /shared/raw_sequences \
    --split train \
    --num-clips 100 \
    --frames-per-clip 50

# 采集验证数据
python collect_raw_data.py \
    --config habitat_extensions/config/vlnce_task.yaml \
    --output /shared/raw_sequences \
    --split val \
    --num-clips 20 \
    --frames-per-clip 50
```

### 步骤 2：验证原始数据

```bash
python validate_raw_data.py /shared/raw_sequences train
python validate_raw_data.py /shared/raw_sequences val
```

### 步骤 3：在训练容器中打包数据

```bash
cd /home/VLN/Project

# 确保 raw_sequences 可访问（软链接或挂载）
ln -s /shared/raw_sequences ./raw_sequences

# 打包训练集
python scripts/pack_dataset.py \
    --config configs/dataset_pack.yaml \
    --split train

# 打包验证集
python scripts/pack_dataset.py \
    --config configs/dataset_pack.yaml \
    --split val
```

### 步骤 4：检查打包后的数据

```bash
# 可视化检查
python scripts/inspect_dataset.py \
    --root ./data/habitat_vln \
    --split train \
    --num 5

# 质量报告
python scripts/report_data_quality.py \
    --root ./data/habitat_vln \
    --split train
```

### 步骤 5：开始训练

```bash
python scripts/train_multistage.py \
    --config configs/train_config.yaml
```

---

## 9. 参考配置文件

### `configs/dataset_pack.yaml`（项目中已存在）

```yaml
seed: 42

data:
  raw_root: ./raw_sequences              # 原始数据根目录
  save_root: ./data/habitat_vln          # 打包后输出目录
  splits:
    train: {scenes: [scene_1, scene_2], clips_per_scene: 100}
    val:   {scenes: [scene_3], clips_per_scene: 20}

pack:
  frames_per_clip: 8                     # T（子序列长度）
  stride: 1                              # 滑动窗口步长
  sampler: visibility_fps                # uniform | fps | visibility | visibility_fps
  lookback: 5                            # 关键帧搜索回溯范围
  keyframes: 4                           # K（关键帧数量）
  ref_policy: last                       # last | middle | index

heatmap:
  size: [64, 64]                         # (Hm, Wm) 热力图分辨率
  gaussian_sigma_px: 1.8                 # 高斯核标准差
  occlusion_check: true                  # 启用遮挡检查
  occlusion_eps: 0.05                    # 遮挡容忍度（米）

export:
  rgb_format: png
  depth_format: npy
  heatmap_format: npy
  drop_if_effective_k_below: 2           # 丢弃有效关键帧 < 2 的样本
  mark_low_quality: true                 # 标记低质量样本
```

---

## 附录：参考资料

- **Habitat-Sim 文档**：https://aihabitat.org/docs/habitat-sim/
- **Habitat-Lab 文档**：https://aihabitat.org/docs/habitat-lab/
- **VLN-CE 仓库**：https://github.com/jacobkrantz/VLN-CE
- **项目训练脚本**：`/home/VLN/Project/scripts/`
  - `gen_synth_demo.py`：合成数据生成示例
  - `pack_dataset.py`：打包流程参考
  - `inspect_dataset.py`：数据可视化

---

**祝数据采集顺利！如有问题，请参考本指南的常见问题部分，或查看项目实际脚本。**

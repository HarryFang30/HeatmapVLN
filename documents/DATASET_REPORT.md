# VLN-CE 双向数据采集完整报告

## 📊 执行摘要

本报告详细记录了基于 **Habitat-Sim** 和 **VLN-CE (Vision-and-Language Navigation in Continuous Environments)** 项目的双向RGB-轨迹数据采集过程。数据采集使用 **Matterport3D** 场景，从 **R2R (Room-to-Room) 数据集**中提取导航任务，采集了包含RGB图像、相机位姿、VLN指令和关键帧匹配的完整训练数据。

**关键指标**:
- **数据集**: VLN-CE-v1 (基于R2R)
- **场景**: Matterport3D (61个室内场景)
- **采集clips**: 100个
- **采样策略**: 双向行走 (forward + backward)
- **数据类型**: RGB图像 + 完整6DOF相机位姿 + VLN元数据

---

## 1. 数据集概述

### 1.1 基础数据集

#### **Matterport3D Dataset**
- **类型**: 大规模3D室内场景数据集
- **场景数量**: 90个完整场景 (本项目使用61个)
- **格式**: `.glb` (GLTF 3D模型)
- **特点**: 
  - 真实世界扫描的室内环境
  - 包含住宅、办公室、酒店等多种类型
  - 高质量纹理和几何结构
- **位置**: `data/scene_datasets/mp3d/`
- **导航网格**: 每个场景配有`.navmesh`文件用于路径规划

**场景示例**:
```
mp3d/
├── 8WUmhLawc2A/
│   ├── 8WUmhLawc2A.glb        # 3D场景模型
│   └── 8WUmhLawc2A.navmesh    # 导航网格
├── JeFG25nYj2p/
├── GdvgFV5R1Z5/
└── ... (共90个场景)
```

#### **R2R (Room-to-Room) Dataset**
- **类型**: Vision-and-Language Navigation 指令数据集
- **原始来源**: Stanford NLP Group
- **格式**: VLN-CE-v1 (Habitat适配版本)
- **文件**: `data/datasets/R2R_VLNCE_v1-3_preprocessed/train/train.json.gz`
- **数据量**:
  - **总episodes**: 10,819个
  - **总场景**: 61个Matterport3D场景
  - **平均episode/场景**: ~177个
- **特点**:
  - 每个episode包含一条自然语言导航指令
  - 起点和终点位置
  - Reference path (关键路径节点)

**数据集统计**:
```json
{
  "total_episodes": 10819,
  "total_scenes": 61,
  "top_scenes": {
    "ur6pFq6Qu1A": 279,
    "8WUmhLawc2A": 279,
    "JeFG25nYj2p": 279,
    "r47D5H71a5s": 279,
    "Vvot9Ly1tCj": 276
  }
}
```

**Episode示例**:
```json
{
  "episode_id": "7677",
  "trajectory_id": "5122",
  "scene_id": "mp3d/8WUmhLawc2A/8WUmhLawc2A.glb",
  "instruction": {
    "instruction_text": "Walk straight toward the kitchen. Stand near the fridge."
  },
  "start_position": [-9.001, 0.101, -2.211],
  "goals": [{"position": [-11.041, 0.101, -7.874]}],
  "reference_path": [
    [-9.001, 0.101, -2.211],
    [-6.850, 0.101, -4.316],
    ...
  ]
}
```

---

## 2. 采集方法

### 2.1 采集策略

#### **双向行走 (Bidirectional Walk)**
为了增加训练数据的多样性和闭环特性，我们采用双向采集策略：

1. **Forward Phase (正向阶段)**:
   - 起点: Episode的`start_position`
   - 终点: Episode的`goals[0].position`
   - 策略: 使用`ShortestPathFollower`沿最短路径前进
   - 最大步数: 50步

2. **Backward Phase (反向阶段)**:
   - 起点: Forward阶段的终点 (goal position)
   - 终点: Episode的`start_position`
   - 策略: 使用`ShortestPathFollower`返回起点
   - 最大步数: 50步

**优势**:
- ✅ 形成闭环轨迹，提供双向视角
- ✅ 增加数据量 (~2倍)
- ✅ 提供反向导航训练样本
- ✅ 更好的场景覆盖

#### **多场景均匀采样**
为避免数据集偏向单一场景，采用轮流采样策略：

```python
# 1. 按场景分组所有episodes
episodes_by_scene = {
    "8WUmhLawc2A": [ep_idx1, ep_idx2, ...],
    "GdvgFV5R1Z5": [ep_idx3, ...],
    ...
}

# 2. 打乱场景顺序
random.seed(42)
random.shuffle(scene_names)

# 3. 轮流从每个场景采样
for each clip:
    scene = scene_names[i % num_scenes]
    episode = random.choice(episodes_by_scene[scene])
```

**结果**: 100个clips覆盖50+个不同场景

### 2.2 采集流程

#### **完整采集管道**:

```
┌─────────────────────────────────────────────────────────────────┐
│ 1. 环境初始化                                                   │
│    ├─ 加载 Habitat Config                                       │
│    ├─ 创建 Habitat Environment                                  │
│    ├─ 计算传感器外参 (T_agent_cam)                             │
│    └─ 创建 ShortestPathFollower                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 2. Episode选择 (多场景均匀采样)                                 │
│    ├─ 从61个场景中轮流选择                                      │
│    ├─ 随机选择该场景的一个episode                               │
│    └─ 提取: instruction, start_pos, goal_pos, reference_path    │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 3. Forward Phase (正向采集)                                     │
│    For each step (max 50):                                      │
│      ├─ 获取 observations (RGB图像)                             │
│      ├─ 计算相机位姿: T_w_c = T_w_agent @ T_agent_cam          │
│      ├─ 保存 RGB: rgb/000000.png, 000001.png, ...              │
│      ├─ 记录轨迹位置: trajectory_positions.append(pos)          │
│      ├─ 获取下一个动作: action = follower.get_next_action(goal)│
│      ├─ 如果 action == STOP: break (避免episode done)           │
│      └─ 执行动作: observations = env.step(action)               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 4. Backward Phase (反向采集)                                    │
│    For each step (max 50):                                      │
│      ├─ 获取 observations (RGB图像)                             │
│      ├─ 计算相机位姿: T_w_c = T_w_agent @ T_agent_cam          │
│      ├─ 保存 RGB: rgb/000051.png, 000052.png, ...              │
│      ├─ 记录轨迹位置: trajectory_positions.append(pos)          │
│      ├─ 获取下一个动作: action = follower.get_next_action(start)│
│      ├─ 如果 action == STOP: break                              │
│      └─ 执行动作: observations = env.step(action)               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 5. 关键帧匹配                                                   │
│    ├─ Forward轨迹 → reference_path 最近邻匹配                   │
│    ├─ Backward轨迹 → reference_path (逆序) 最近邻匹配           │
│    └─ 计算匹配距离 (用于质量评估)                               │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ 6. 数据保存                                                     │
│    ├─ rgb/*.png              (RGB图像序列)                      │
│    ├─ poses.json             (相机位姿: 4×4矩阵)               │
│    ├─ intrinsics.json        (相机内参: K矩阵)                 │
│    └─ meta.json              (VLN元数据)                        │
└─────────────────────────────────────────────────────────────────┘
```

### 2.3 核心技术细节

#### **相机位姿计算**

完整的6DOF相机位姿通过以下变换链计算：

```
World → Agent → Camera

T_w_c = T_w_agent @ T_agent_cam
```

**步骤**:
1. **获取Agent状态**:
   ```python
   agent_state = sim.get_agent_state()
   position = agent_state.position      # [x, y, z]
   rotation = agent_state.rotation      # Quaternion
   ```

2. **Quaternion → 旋转矩阵**:
   ```python
   def quaternion_to_rotation_matrix(q):
       w, x, y, z = q.scalar, q.vector.x, q.vector.y, q.vector.z
       R = [
           [1-2*(y²+z²), 2*(xy-wz), 2*(xz+wy)],
           [2*(xy+wz), 1-2*(x²+z²), 2*(yz-wx)],
           [2*(xz-wy), 2*(yz+wx), 1-2*(x²+y²)]
       ]
       return R
   ```

3. **构造Agent-to-World变换**:
   ```python
   T_w_agent = [
       [R_agent, position],
       [0, 0, 0, 1]
   ]
   ```

4. **应用传感器外参**:
   ```python
   T_agent_cam = get_sensor_extrinsics(config)  # 从配置读取
   T_w_c = T_w_agent @ T_agent_cam
   ```

**传感器外参** (默认配置):
```python
# RGB_SENSOR配置
POSITION: [0.0, 1.25, 0.0]  # 相机相对Agent位置 (高度1.25m)
ORIENTATION: [0, 0, 0]       # 无额外旋转
```

#### **相机内参计算**

根据Habitat配置计算针孔相机模型参数：

```python
def compute_intrinsics(config):
    width = config.SIMULATOR.RGB_SENSOR.WIDTH      # 224
    height = config.SIMULATOR.RGB_SENSOR.HEIGHT    # 224
    hfov = config.SIMULATOR.RGB_SENSOR.HFOV        # 90度
    
    # 焦距公式
    fx = width / (2 * tan(hfov/2))
    fy = fx
    cx = width / 2
    cy = height / 2
    
    # 内参矩阵
    K = [
        [fx,  0, cx],
        [ 0, fy, cy],
        [ 0,  0,  1]
    ]
```

**默认配置**:
- 分辨率: 224×224
- HFOV: 90°
- fx = fy = 112.0
- cx = cy = 112.0

#### **关键帧匹配算法**

将采集的轨迹与R2R的reference_path对齐：

```python
def match_keyframes_to_trajectory(trajectory, reference_path):
    keyframe_indices = []
    keyframe_distances = []
    
    for ref_point in reference_path:
        # 找到轨迹中最接近的帧
        distances = [||traj_pos - ref_point|| for traj_pos in trajectory]
        closest_idx = argmin(distances)
        
        keyframe_indices.append(closest_idx)
        keyframe_distances.append(distances[closest_idx])
    
    return keyframe_indices, keyframe_distances
```

**质量指标**:
- `max_keyframe_distance`: 最大匹配误差
- `mean_keyframe_distance`: 平均匹配误差
- 用于筛选高质量轨迹

---

## 3. 采集的数据

### 3.1 数据结构

#### **完整目录树**:

```
raw_sequences_working/
├── train/                           # 训练集数据
│   ├── 8WUmhLawc2A/                # 场景1
│   │   ├── clip_000001/
│   │   │   ├── rgb/                 # RGB图像序列
│   │   │   │   ├── 000000.png      # Forward帧0
│   │   │   │   ├── 000001.png      # Forward帧1
│   │   │   │   ├── ...
│   │   │   │   ├── 000049.png      # Forward帧49
│   │   │   │   ├── 000050.png      # Backward帧0
│   │   │   │   └── ...
│   │   │   ├── poses.json           # 相机位姿 (N×4×4)
│   │   │   ├── intrinsics.json      # 相机内参
│   │   │   └── meta.json            # VLN元数据
│   │   └── clip_000014/
│   │       └── ...
│   ├── GdvgFV5R1Z5/                # 场景2
│   │   └── clip_000002/
│   ├── JF19kD82Mey/                # 场景3
│   │   └── clip_000003/
│   ├── ... (共61个场景)
│   └── ur6pFq6Qu1A/                # 场景N
│       └── clip_000004/
├── progress.json                    # 采集进度
├── collection_stats.json            # 统计信息
└── train_index.json                # 全局索引
```

### 3.2 数据文件详解

#### **1. RGB图像 (`rgb/*.png`)**

**格式**:
- 图像格式: PNG
- 分辨率: 224×224
- 通道: 3 (BGR)
- 位深: 8-bit
- 文件大小: 65-86 KB/张

**命名规则**:
```
000000.png  # Forward第0帧
000001.png  # Forward第1帧
...
000049.png  # Forward第49帧 (如果到达)
000050.png  # Backward第0帧
...
```

**示例图像特征**:
- 真实室内场景纹理
- 家具、墙壁、装饰物等
- 自然光照效果

#### **2. 相机位姿 (`poses.json`)**

**格式**: JSON数组，每个元素为4×4变换矩阵

```json
[
  [
    [0.5000, 0.0000, -0.8660, 10.2578],    // 行1: [R11, R12, R13, tx]
    [0.0000, 1.0000,  0.0000,  1.3436],    // 行2: [R21, R22, R23, ty]
    [0.8660, 0.0000,  0.5000, -2.3797],    // 行3: [R31, R32, R33, tz]
    [0.0000, 0.0000,  0.0000,  1.0000]     // 行4: [0, 0, 0, 1]
  ],
  [ ... ]  // 下一帧
]
```

**矩阵结构**:
```
T_w_c = [R | t]
        [0 | 1]

R: 3×3 旋转矩阵 (相机方向)
t: 3×1 平移向量 (相机位置, 单位: 米)
```

**坐标系**:
- Origin: Matterport3D场景坐标系原点
- X轴: 向右
- Y轴: 向上
- Z轴: 向前 (右手系)

#### **3. 相机内参 (`intrinsics.json`)**

```json
{
  "fx": 112.0,                    // X轴焦距 (像素)
  "fy": 112.0,                    // Y轴焦距 (像素)
  "cx": 112.0,                    // 主点X坐标 (像素)
  "cy": 112.0,                    // 主点Y坐标 (像素)
  "K": [                          // 内参矩阵 (3×3)
    [112.0, 0.0, 112.0],
    [0.0, 112.0, 112.0],
    [0.0, 0.0, 1.0]
  ],
  "width": 224,                   // 图像宽度
  "height": 224,                  // 图像高度
  "hfov": 90                      // 水平视场角 (度)
}
```

**用途**:
- 3D点投影到2D图像
- RGB-D配准
- 相机标定

#### **4. VLN元数据 (`meta.json`)**

```json
{
  // ========== 基础信息 ==========
  "episode_id": "7677",                    // VLN-CE episode ID
  "trajectory_id": "5122",                 // R2R trajectory ID
  "scene_id": "8WUmhLawc2A",              // Matterport3D场景名
  
  // ========== VLN核心数据 ==========
  "instruction": "Walk straight toward the kitchen. Stand near the fridge.",
  
  "reference_path": [                      // R2R关键路径节点
    [-9.001, 0.101, -2.211],              // 节点1: [x, y, z]
    [-6.850, 0.101, -4.316],              // 节点2
    [-6.427, 0.101, -5.572],              // 节点3
    // ... 更多节点
  ],
  
  // ========== 采集策略 ==========
  "sampling_strategy": "bidirectional_walk",
  "num_frames": 51,                        // 总帧数
  
  // ========== 正向段信息 ==========
  "forward_segment": {
    "start_frame": 0,                      // 起始帧索引
    "end_frame": 49,                       // 结束帧索引
    "num_frames": 50                       // 帧数
  },
  
  // ========== 反向段信息 ==========
  "backward_segment": {
    "start_frame": 50,
    "end_frame": 50,
    "num_frames": 1
  },
  
  // ========== 关键帧匹配 ==========
  "forward_keyframe_indices": [48, 48, 48, 48, 48, 48, 48],
  "forward_keyframe_distances": [
    12.629, 11.106, 11.256, 13.135, 13.931, 14.914, 16.410
  ],
  "backward_keyframe_indices": [50, 50, 50, 50, 50, 50, 50],
  "backward_keyframe_distances": [
    16.167, 14.668, 13.687, 12.893, 11.013, 10.858, 12.381
  ],
  
  // ========== 质量指标 ==========
  "forward_max_keyframe_distance": 16.410,   // 正向最大匹配误差 (米)
  "forward_mean_keyframe_distance": 13.340,  // 正向平均匹配误差
  "backward_max_keyframe_distance": 16.167,
  "backward_mean_keyframe_distance": 13.095
}
```

**关键字段说明**:

| 字段 | 类型 | 用途 |
|------|------|------|
| `instruction` | String | VLN任务的自然语言指令 |
| `reference_path` | Array[Array[3]] | R2R数据集的标准路径节点 |
| `forward/backward_keyframe_indices` | Array[Int] | 每个reference节点对应的帧索引 |
| `forward/backward_keyframe_distances` | Array[Float] | 匹配误差 (用于质量筛选) |

#### **5. 全局索引 (`train_index.json`)**

包含所有clips的meta信息列表，便于快速检索：

```json
[
  {
    "episode_id": "7677",
    "trajectory_id": "5122",
    "scene_id": "8WUmhLawc2A",
    "instruction": "Walk straight toward...",
    "num_frames": 51,
    // ... (完整meta数据)
  },
  {
    "episode_id": "9557",
    "trajectory_id": "6234",
    "scene_id": "GdvgFV5R1Z5",
    // ...
  }
  // ... (100个clips)
]
```

#### **6. 采集统计 (`collection_stats.json`)**

```json
{
  "successful": 100,                       // 成功采集的clips
  "failed": 0,                             // 失败的clips
  "failed_clips": [],                      // 失败详情
  "scenes": {                              // 场景分布
    "8WUmhLawc2A": 2,
    "GdvgFV5R1Z5": 1,
    // ... (共50+个场景)
  },
  "total_frames": 9500                     // 总帧数 (约)
}
```

#### **7. 进度文件 (`progress.json`)**

支持断点续传：

```json
{
  "last_completed_clip": 45               // 最后完成的clip ID
}
```


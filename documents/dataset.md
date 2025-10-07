# 数据集接口与打包实施手册（dataset.md｜**无外部仿真依赖版**）

> 这份手册是给 Claude 的 **一步一步可实现清单**。目标：在**不依赖任何仿真/环境**的情况下，完成：
> 1) **确定性关键帧采样器**（Uniform/FPS）；
> 2) **几何投影与热力图生成**（基于给定 `RGB/Depth/位姿/内参` 的纯数学实现）；
> 3) **数据打包脚本**（把已有/外部产出的原始序列打包成标准训练集）；
> 4) **读取适配器** 与 **质检工具**；
> 5) **合成 Demo**（无外部依赖，直接验证全链路）。
>
> 你后续无论用什么方式产原始 `RGB/Depth/位姿/内参`，只要格式对齐，这套代码都能直接跑；训练脚本按 `train.md` 的接口吃即可。

---

## 0) 目录与文件（Claude 逐条创建/修改）
```
configs/
  └─ dataset_pack.yaml               # ✨ 打包与采样配置（与来源无关）

scripts/
  ├─ pack_dataset.py                 # ✨ 将已有原始序列 → 标准训练集
  ├─ inspect_dataset.py              # ✨ 质检：热力图叠加可视化
  └─ gen_synth_demo.py               # ✨ 生成一个不依赖外部的合成 Demo（自证代码可跑）

src/
  └─ data/
     ├─ heatmap_builder.py           # ✨ 纯数学/几何：内参、投影、FPS、热力图
     └─ vln_heatmap_adapter.py       # ✨ DataLoader：返回 train.md 期望的 batch

models/                                # ✅ 保持（HF/VGGT 等权重）
```

---

## 1) 标准数据格式（输出成这样，训练直接可用）
```
<data_root>/
  train/ | val/ | test/
    <scene_id>/
      clip_000001/
        rgb/000000.png ... 000{T-1}.png
        depth/000000.npy ... 000{T-1}.npy       # float32, 与 RGB 对齐
        poses.json                               # 列表长度 T；每项 4x4 行主序 T_world_cam
        intrinsics.json                          # fx/fy/cx/cy 或 3x3 K；可单一/逐帧
        heatmaps.npy                             # [K, Hm, Wm] float32，sum≈1（每张）
        mask.npy                                 # [K] float32（全零→0，否则1）
        meta.json                                # ref_idx/key_indices/采样法/尺寸等
```

**meta.json 示例**：
```json
{
  "scene": "RoomA",
  "episode_id": 12,
  "T": 8,
  "K": 4,
  "ref_idx": 7,
  "key_indices": [0, 2, 4, 6],
  "sampler": {"type": "fps", "alpha": 1.0, "beta": 0.7, "seed": 42},
  "image_size": [384, 384],
  "heatmap_size": [64, 64],
  "gaussian_sigma_px": 3.0,
  "occlusion_check": true,
  "occlusion_eps": 0.05
}
```

---

## 2) 配置模板（`configs/dataset_pack.yaml`）
```yaml
seed: 42

data:
  raw_root: ./raw_sequences                  # 你已有的原始序列目录（见 §3 输入约定）
  save_root: ./data/habitat_vln              # 打包后的标准训练集输出根
  splits:
    train: {scenes: [RoomA], clips_per_scene: 8}
    val:   {scenes: [RoomB], clips_per_scene: 2}
    test:  {scenes: [RoomC], clips_per_scene: 2}

pack:
  frames_per_clip: 8                         # T
  stride: 1                                  # 取帧步长（从原始序列下采样）
  sampler: fps                               # uniform | fps
  keyframes: 4                               # K
  fps_alpha: 1.0                             # 位姿距离：平移权重
  fps_beta: 0.7                              # 位姿距离：朝向权重
  ref_policy: last                           # last | middle | index
  ref_index: -1

heatmap:
  size: [64, 64]                             # (Hm, Wm)
  gaussian_sigma_px: 3.0
  occlusion_check: true
  occlusion_eps: 0.05

export:
  rgb_format: png
  depth_format: npy
  heatmap_format: npy
```

---

## 3) 原始序列输入约定（**不限定来源**）
> 只要能满足下列约定，`scripts/pack_dataset.py` 就能把它打包成标准训练集：

```
raw_sequences/
  train/ | val/ | test/
    <scene_id>/
      clip_xxxxxx/
        rgb/000000.png ...              # T_raw 帧
        depth/000000.npy ...            # 与 RGB 对齐
        poses.json                      # 列表长度 T_raw，每项为 4x4 T_world_cam
        intrinsics.json                 # 单一/逐帧 K 或 {fx,fy,cx,cy}
```
- 若暂时没有原始序列，可先跑 `scripts/gen_synth_demo.py` 生成一个**合成小数据**，验证代码链路。

---

## 4) 几何工具（`src/data/heatmap_builder.py`）
**Claude：实现以下函数签名（纯 numpy/cv2/torch，禁止外部依赖）**

### 4.1 相机内参
```python
def build_intrinsics(width: int, height: int, fx: float | None = None, fy: float | None = None,
                     cx: float | None = None, cy: float | None = None, hfov_deg: float | None = None) -> dict:
    """根据已知 fx/fy/cx/cy 或 hfov 推算；返回 {K, Kinv, fx, fy, cx, cy}。"""
```

### 4.2 SO(3)/SE(3) 度量（FPS 用）
```python
def so3_angle(Ra: np.ndarray, Rb: np.ndarray) -> float: ...

def se3_pose_distance(Ta: np.ndarray, Tb: np.ndarray, alpha: float, beta: float) -> float: ...
```

### 4.3 关键帧采样器（确定性）
```python
def uniform_keyframe_indices(T: int, K: int, ref_idx: int) -> list[int]: ...

def fps_keyframe_indices(poses: list[np.ndarray], K: int, ref_idx: int,
                          alpha: float, beta: float, seed: int) -> list[int]: ...
```

### 4.4 回投/投影（像素↔相机↔世界）
```python
def unproject_depth_to_points(depth: np.ndarray, Kinv: np.ndarray) -> np.ndarray: ...

def world_from_cam(points_c: np.ndarray, T_w_c: np.ndarray) -> np.ndarray: ...

def cam_from_world(points_w: np.ndarray, T_c_w: np.ndarray) -> np.ndarray: ...

def project_cam_points_to_pixels(points_i: np.ndarray, fx: float, fy: float, cx: float, cy: float,
                                 w: int, h: int) -> tuple[np.ndarray, np.ndarray]:
    """返回 (uv[N,2], z[N])，过滤 z<=0 & 越界。"""
```

### 4.5 遮挡过滤（可选）
```python
def occlusion_filter(uv: np.ndarray, z: np.ndarray, depth_ref: np.ndarray, eps: float) -> np.ndarray: ...
```

### 4.6 关键帧 → 参考帧（整合）
```python
def project_keyframe_to_ref(depth_j: np.ndarray, Kj: dict, T_w_c_j: np.ndarray,
                            T_c_ref_w: np.ndarray, depth_ref: np.ndarray | None,
                            ref_w: int, ref_h: int, occl_eps: float | None) -> tuple[np.ndarray, np.ndarray]:
    """返回 (uv_ref[N,2], valid_mask[N])。"""
```

### 4.7 热力图生成
```python
def heatmap_from_points(uv_ref: np.ndarray, ref_size: tuple[int,int], hm_size: tuple[int,int],
                        sigma_px: float) -> np.ndarray:
    """返回 [Hm,Wm] float32 概率热力图（sum=1；无点则全0）。"""
```

> 注意点：
> - 内参支持两种输入路径（显式 fx/fy/cx/cy 或 hfov 反推）；
> - 投影时返回 z 以便做遮挡过滤；
> - 热力图先散点计数，再高斯平滑，再下采样，再归一化。

---

## 5) 打包脚本（`scripts/pack_dataset.py`）
**Claude：实现下列流程（与 `configs/dataset_pack.yaml` 对齐）**

1. 读取配置，设随机种子，创建输出根 `<save_root>/<split>/<scene>/clip_xxxxxx/`。
2. 遍历 `raw_root/<split>/<scene>/clip_xxxxxx/`：
   - 读取 `rgb/`、`depth/`、`poses.json`、`intrinsics.json`；
   - 以 `frames_per_clip` + `stride` 切出子序列（子窗口）；
   - 按 `ref_policy` 选 `ref_idx`（默认最后一帧）；
   - 从 `poses` 中取前 `ref_idx` 的外参，使用 `uniform_keyframe_indices` 或 `fps_keyframe_indices` 选 K 个关键帧。
3. 对每个关键帧 j：
   - 组装 `Kj = build_intrinsics(... from intrinsics.json ...)`；
   - 计算 `T_c_ref_w = inv(T_w_c[ref_idx])`；
   - `uv,valid = project_keyframe_to_ref(depth[j], Kj, T_w_c[j], T_c_ref_w, depth_ref=depth[ref_idx] if occlusion_check else None, ...)`；
   - `hm_j = heatmap_from_points(uv[valid], (H,W), (Hm,Wm), sigma_px)`。
4. 叠成 `heatmaps.npy`（[K,Hm,Wm]）与 `mask.npy`（[K]，全零→0）；写入 `meta.json`；将对应的 `rgb/`、`depth/`、`poses.json`、`intrinsics.json` 一并拷贝/裁剪后保存。
5. 每 50 个 clip 打印一次进度与空热图比例；每个 scene 输出统计。

**命令**：
```bash
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split train
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split val
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split test
```

---

## 6) 读取适配器（`src/data/vln_heatmap_adapter.py`）
**Claude：实现 DataLoader，返回 train.md 期望字段**
```python
class VLNHeatmapDataset(torch.utils.data.Dataset):
    def __init__(self, root: str, split: str,
                 frames_per_clip: int, heatmap_per_clip: int,
                 image_size=(384,384), hm_size=(64,64)):
        # 枚举 <root>/<split>/<scene>/clip_xxxxxx/
        ...
    def __getitem__(self, idx):
        # 读取 rgb/ depth/ poses.json/ intrinsics.json/ heatmaps.npy/ mask.npy/ meta.json
        # RGB → Tensor[T,3,H,W]；Heatmaps → Tensor[K,Hm,Wm]（与 hm_size 不同则插值+归一）
        return {"frames": frames, "text": None, "gt_heatmaps": hms, "mask": mask, "meta": meta}
    def __len__(self): ...
```

---

## 7) 质检工具（`scripts/inspect_dataset.py`）
**Claude：实现快速可视化**
- 将 `heatmaps[k]` 上采样到 RGB 分辨率并着色叠加在参考帧；
- 输出到 `outputs/inspect/<scene>/clip_xxxxxx_overlay.png`；
- 打印空热图比例。

```bash
python scripts/inspect_dataset.py --root ./data/habitat_vln --split train --num 8
```

---

## 8) 合成 Demo（`scripts/gen_synth_demo.py`）
> 无外部依赖地“自证链路可跑”。随机生成一组 **一致的** `poses/intrinsics` 与 **可控的** `depth`，让关键帧投影到参考帧时确实能看到热点。

**建议实现**：
1. 构造一个 3D 平面/棋盘点云；
2. 生成 T 个相机位姿（绕场景环转/平移），保证参考帧能“看到”前面帧的点；
3. 用真内参渲染“深度图”（把点投影回去生成稀疏深度，再高斯扩散成密深度）；
4. 保存为 `raw_sequences/train/RoomA/clip_000001/...`；
5. 立刻运行 `pack_dataset.py` → `inspect_dataset.py` 检查叠加图是否合理。

命令：
```bash
python scripts/gen_synth_demo.py --root ./raw_sequences --scene RoomA --clips 1 --T 8 --W 384 --H 384
python scripts/pack_dataset.py  --config configs/dataset_pack.yaml --split train
python scripts/inspect_dataset.py --root ./data/habitat_vln --split train --num 1
```

---

## 9) 与训练衔接
- 在训练配置里把 `data.root` 指向 `./data/habitat_vln`；
- 训练脚本会吃到 `frames[T,3,H,W]`、`gt_heatmaps[K,Hm,Wm]`、`mask[K]`、`meta` 四件套，直接开跑。

---

## 10) Claude 的待办 Checklist（一次过完）
- [ ] 新建/更新 §0 的 5 个文件/脚本/配置。
- [ ] 完成 `src/data/heatmap_builder.py` **全部函数**（有基本单测/断言）。
- [ ] 写好 `scripts/gen_synth_demo.py`，生成一条合成原始序列到 `raw_sequences/`。
- [ ] 跑 `scripts/pack_dataset.py` 将合成序列打包到标准训练集目录。
- [ ] 跑 `scripts/inspect_dataset.py` 输出叠加图，确认热点位置合理、空热图比例不高（<20%）。
- [ ] 用 `VLNHeatmapDataset` 读取一个 batch，核对张量形状与字段名与 `train.md` 完全一致。

> 完成本清单后，**无需任何外部仿真/环境**，你的代码就具备：确定性采样、几何投影、热力图生成、数据打包、读取与质检的全链路能力。等你之后把真实序列产出放进 `raw_sequences/`，只需再跑一次 `pack_dataset.py` 即可对接训练。


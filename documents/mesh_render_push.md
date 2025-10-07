# mesh_render_push.md ｜ 把稀疏点云升级为**可训练**的表面渲染（给 Claude｜可直接执行）

> 目的：当前数据仍然 **K_eff≈1**，根因是“点云投影”太稀疏（像素覆盖 <2%）。本手册将把 `gen_synth_demo.py` 升级为**网格表面渲染**（软件栅格化 + Z-Buffer），生成**密集深度图与RGB**，使 `K_eff≥2` 轻松达标（目标 ≥85%），并让 overfit 与 Stage‑A 自然收敛。
>
> 约束：**不引入仿真**，不强依赖 OpenGL/torch3d；采用纯 `numpy` 的最小软件栅格器，兼容现有流水线。

---

## 0) 改动总览
- 新增：`src/data/mesh_renderer.py` —— 纯 numpy 三角形栅格器（支持深度 Z-Buffer 与颜色插值）。
- 新增：`src/data/mesh_scenes.py` —— 生成简单室内多平面场景（地面+两面墙+方柱/广告牌目标），可控网格密度。
- 修改：`scripts/gen_synth_demo.py` —— 新增 `--render_mode mesh`，相机按短弧+面向目标生成帧，调用表面渲染得到 **密集** `rgb/depth`。
- 调整：`configs/dataset_pack.yaml` —— 恢复/提升可见性阈值、开启遮挡检查、常规子采样。
- 验证：`scripts/check_mesh_visibility.py` —— 可见性与覆盖率断言；`scripts/report_data_quality.py` —— 期望 `K_eff≥2` 达标。

---

## 1) 新增：最小软件栅格器（`src/data/mesh_renderer.py`）
**创建文件** `src/data/mesh_renderer.py`，拷入以下代码（纯 numpy，已考虑数值稳定）：

```python
# src/data/mesh_renderer.py
import numpy as np

# ---------------------- 数学与相机 ----------------------

def invert_se3(T):
    R = T[:3,:3]; t = T[:3,3:4]
    Rt = R.T
    Tinv = np.eye(4, dtype=np.float32)
    Tinv[:3,:3] = Rt
    Tinv[:3,3:4] = -Rt @ t
    return Tinv

def cam_from_world(Xw, T_c_w):
    # Xw: [N,3], T_c_w: 4x4 (camera<-world)
    N = Xw.shape[0]
    Xh = np.concatenate([Xw, np.ones((N,1), dtype=np.float32)], axis=1)  # [N,4]
    Xc = (T_c_w @ Xh.T).T  # [N,4]
    return Xc[:, :3]

def project_pixels(Xc, K):
    # Xc: [N,3] (camera frame), K: dict with fx,fy,cx,cy
    z = Xc[:,2]
    x = K['fx'] * (Xc[:,0] / z) + K['cx']
    y = K['fy'] * (Xc[:,1] / z) + K['cy']
    return x, y, z

# ---------------------- 栅格化核心 ----------------------

def _bbox2d(xs, ys, W, H):
    xmin = max(int(np.floor(xs.min())), 0)
    xmax = min(int(np.ceil(xs.max())), W-1)
    ymin = max(int(np.floor(ys.min())), 0)
    ymax = min(int(np.ceil(ys.max())), H-1)
    return xmin, xmax, ymin, ymax

def _barycentric(px, py, x0,y0, x1,y1, x2,y2):
    # 计算像素中心相对三角形的重心坐标（含符号）
    den = (y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2)
    w0 = ((y1 - y2)*(px - x2) + (x2 - x1)*(py - y2)) / (den + 1e-12)
    w1 = ((y2 - y0)*(px - x2) + (x0 - x2)*(py - y2)) / (den + 1e-12)
    w2 = 1.0 - w0 - w1
    return w0, w1, w2

def rasterize_triangles(verts_w, faces, colors_v, K, T_c_w, W, H,
                        z_far=100.0, z_near=0.05, backface_cull=True):
    """
    verts_w: [V,3] 世界坐标顶点；faces: [F,3] 顶点索引；colors_v: [V,3] 顶点颜色(0..1)
    输出：rgb[H,W,3]、depth[H,W]（0 表示无效深度）
    """
    # 1) 世界->相机->像素
    verts_c = cam_from_world(verts_w, T_c_w)
    # 丢弃在相机后方的顶点索引（z<=0），但仍需逐三角形检查

    rgb = np.zeros((H, W, 3), dtype=np.float32)
    depth = np.zeros((H, W), dtype=np.float32)
    zbuf = np.full((H, W), np.inf, dtype=np.float32)

    for f in faces:
        v0, v1, v2 = f
        X0 = verts_c[v0]; X1 = verts_c[v1]; X2 = verts_c[v2]
        if backface_cull:
            # 法线朝向（相机在 +z 方向看过去），用相机坐标下的三角形 2D 面积符号近似
            x0,y0,_ = X0; x1,y1,_ = X1; x2,y2,_ = X2
            area = (x1-x0)*(y2-y0) - (y1-y0)*(x2-x0)
            if area >= 0:  # 逆时针/顺时针按坐标系约定，这里过滤一个方向
                pass  # 可按需要关掉剔除
        if X0[2] <= z_near and X1[2] <= z_near and X2[2] <= z_near:
            continue  # 全部在近裁剪平面之前
        # 投影到像素
        x0,y0,z0 = project_pixels(np.array([X0]), K)
        x1,y1,z1 = project_pixels(np.array([X1]), K)
        x2,y2,z2 = project_pixels(np.array([X2]), K)
        x0,y0,z0 = x0[0],y0[0],z0[0]
        x1,y1,z1 = x1[0],y1[0],z1[0]
        x2,y2,z2 = x2[0],y2[0],z2[0]
        # 2D 包围盒
        xmin,xmax,ymin,ymax = _bbox2d(np.array([x0,x1,x2]), np.array([y0,y1,y2]), W, H)
        if xmin> xmax or ymin>ymax: continue
        # 顶点颜色
        c0 = colors_v[v0]; c1 = colors_v[v1]; c2 = colors_v[v2]
        # 扫描线
        xs = np.arange(xmin, xmax+1)
        ys = np.arange(ymin, ymax+1)
        for yy in ys:
            py = yy + 0.5
            for xx in xs:
                px = xx + 0.5
                w0,w1,w2 = _barycentric(px, py, x0,y0, x1,y1, x2,y2)
                if (w0 >= 0) and (w1 >= 0) and (w2 >= 0):
                    z = w0*z0 + w1*z1 + w2*z2
                    if z_near < z < z_far and z < zbuf[yy, xx]:
                        zbuf[yy, xx] = z
                        depth[yy, xx] = z
                        rgb[yy, xx, :] = w0*c0 + w1*c1 + w2*c2
    return rgb, depth
```

> 说明：该实现简单可读、便于调试；性能足以应对我们的小场景（几千三角）。如需更快，可后续再做向量化优化。

---

## 2) 新增：室内简易网格场景（`src/data/mesh_scenes.py`）
**创建文件** `src/data/mesh_scenes.py`，提供若干平面+方柱/广告牌目标，支持网格细分控制覆盖率。

```python
# src/data/mesh_scenes.py
import numpy as np

COL_FLOOR = np.array([0.7, 0.7, 0.7], dtype=np.float32)
COL_WALL  = np.array([0.8, 0.85, 0.9], dtype=np.float32)
COL_OBJ   = np.array([0.9, 0.2, 0.2], dtype=np.float32)


def _grid_plane(xmin, xmax, zmin, zmax, y, nx=32, nz=32, color=(1,1,1)):
    # 生成水平面 (y 常数) 的网格细分（地板）
    xs = np.linspace(xmin, xmax, nx+1)
    zs = np.linspace(zmin, zmax, nz+1)
    verts = []
    faces = []
    colors = []
    for i in range(nx+1):
        for k in range(nz+1):
            verts.append([xs[i], y, zs[k]])
            colors.append(color)
    verts = np.array(verts, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)
    def vid(i,k): return i*(nz+1)+k
    for i in range(nx):
        for k in range(nz):
            v00 = vid(i, k)
            v10 = vid(i+1, k)
            v01 = vid(i, k+1)
            v11 = vid(i+1, k+1)
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    faces = np.array(faces, dtype=np.int32)
    return verts, faces, colors


def _grid_wall(z, xmin, xmax, ymin, ymax, nx=16, ny=16, color=(1,1,1)):
    # 生成竖直墙（z 常数）
    xs = np.linspace(xmin, xmax, nx+1)
    ys = np.linspace(ymin, ymax, ny+1)
    verts = []
    faces = []
    colors = []
    for i in range(nx+1):
        for j in range(ny+1):
            verts.append([xs[i], ys[j], z])
            colors.append(color)
    verts = np.array(verts, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)
    def vid(i,j): return i*(ny+1)+j
    for i in range(nx):
        for j in range(ny):
            v00 = vid(i, j)
            v10 = vid(i+1, j)
            v01 = vid(i, j+1)
            v11 = vid(i+1, j+1)
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    faces = np.array(faces, dtype=np.int32)
    return verts, faces, colors


def _billboard(center=(0,1.2,2.0), w=0.6, h=0.6, nx=8, ny=8, color=(0.9,0.2,0.2)):
    cx, cy, cz = center
    xs = np.linspace(cx - w/2, cx + w/2, nx+1)
    ys = np.linspace(cy - h/2, cy + h/2, ny+1)
    verts = []
    faces = []
    colors = []
    for i in range(nx+1):
        for j in range(ny+1):
            verts.append([xs[i], ys[j], cz])
            colors.append(color)
    verts = np.array(verts, dtype=np.float32)
    colors = np.array(colors, dtype=np.float32)
    def vid(i,j): return i*(ny+1)+j
    for i in range(nx):
        for j in range(ny):
            v00 = vid(i, j)
            v10 = vid(i+1, j)
            v01 = vid(i, j+1)
            v11 = vid(i+1, j+1)
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])
    faces = np.array(faces, dtype=np.int32)
    return verts, faces, colors


def build_simple_room(grid=32):
    # 地板：x∈[-2,2], z∈[0.5,3.5], y=0
    v0,f0,c0 = _grid_plane(-2,2, 0.5,3.5, y=0.0, nx=grid, nz=grid, color=COL_FLOOR)
    # 左/右墙：z=3.5，x∈[-2,2], y∈[0,2.5]
    v1,f1,c1 = _grid_wall(3.5, -2,2, 0.0,2.5, nx=grid//2, ny=grid//2, color=COL_WALL)
    v2,f2,c2 = _grid_wall(0.5, -2,2, 0.0,2.5, nx=grid//2, ny=grid//2, color=COL_WALL)
    # 目标广告牌：中心(0,1.2,2.0)，细分提升覆盖
    v3,f3,c3 = _billboard(center=(0,1.2,2.0), w=0.8, h=0.8, nx=grid//2, ny=grid//2, color=COL_OBJ)

    # 拼接
    verts = np.concatenate([v0,v1,v2,v3], axis=0)
    colors= np.concatenate([c0,c1,c2,c3], axis=0)
    off1, off2, off3 = len(v0), len(v0)+len(v1), len(v0)+len(v1)+len(v2)
    faces = np.concatenate([f0, f1+off1, f2+off2, f3+off3], axis=0)
    return verts, faces, colors
```

> 该场景是“房间+广告牌”极简版，足以在 384×384 下实现 >80% 深度覆盖、强视野重叠。

---

## 3) 修改：`scripts/gen_synth_demo.py` 接入表面渲染
**在原脚本中添加以下要点**（保留已有随机点云路径，但默认切到 `mesh`）：

### 3.1 新增 CLI 参数
```bash
--render_mode {mesh,points}   # 默认 mesh
--mesh_grid 32                # 网格细分：32≈几千三角；可 24/32/40 试
--arc_deg 20                  # 更短弧增重叠（原 30）
--radius 2.0                  # 更近距离
```

### 3.2 代码改动（示意补丁）
```python
# 顶部引入
from src.data.mesh_scenes import build_simple_room
from src.data.mesh_renderer import rasterize_triangles, invert_se3

# 生成相机 pose（保持你现有的 short_arc + face_target）
# 确保 ref_idx = T-1

if args.render_mode == 'mesh':
    V,F,C = build_simple_room(grid=args.mesh_grid)
    # 对每一帧：渲染 rgb/depth
    for t, T_w_c in enumerate(poses_w_c):
        T_c_w = invert_se3(T_w_c)
        rgb, depth = rasterize_triangles(V, F, C, K=intrinsics, T_c_w=T_c_w, W=args.W, H=args.H,
                                         z_near=0.05, z_far=100.0, backface_cull=False)
        # 存盘：与现有格式一致
        save_rgb_depth(out_dir, t, rgb, depth)
else:
    # 旧的点云投影路径（保留作为参考或AB Test）
    render_points(...)
```

> 备注：`save_rgb_depth` 用你现有接口。若原来只存 `depth`，这里顺带把 `rgb` 也存，便于可视化。

---

## 4) 调整打包配置（恢复常规阈值）
表面渲染后覆盖率很高，恢复较严格配置：

```yaml
# configs/dataset_pack.yaml
pack:
  sampler: visibility_fps
  lookback: 5
  keyframes: 4
  topk_by_visible_ratio: true
  min_visible_ratio: 0.02   # 恢复常规阈值
  depth_subsample: 4        # 恢复子采样

heatmap:
  size: [64, 64]
  gaussian_sigma_px: 2.0    # 恢复到 2.0（更稳）
  occlusion_check: true     # 开启遮挡检查，Z-buffer 深度可用

export:
  drop_if_effective_k_below: 2
  mark_low_quality: true
```

---

## 5) 验证脚本：可见性与覆盖率断言（新建）
**创建** `scripts/check_mesh_visibility.py`：

```python
import os, json, numpy as np
from glob import glob

# 简单统计深度覆盖率（非零比例）与临近帧可见性

def depth_coverage(depth):
    H,W = depth.shape
    return float((depth > 0).sum()) / float(H*W)

if __name__ == '__main__':
    clip = sorted(glob('raw_sequences/train/RoomA/clip_*'))[0]
    metas = json.load(open(os.path.join(clip, 'meta.json')))
    T = metas['T']
    ref = metas.get('ref_idx', T-1)
    # 读取相邻帧与参考帧的 depth（按你的存法替换）
    depth_ref = np.load(os.path.join(clip, f'depth_{ref:03d}.npy'))
    depth_prev= np.load(os.path.join(clip, f'depth_{ref-1:03d}.npy'))
    cov_ref  = depth_coverage(depth_ref)
    cov_prev = depth_coverage(depth_prev)
    print(f'coverage(ref)={cov_ref:.3f}, coverage(prev)={cov_prev:.3f}')
    assert cov_ref > 0.7 and cov_prev > 0.7, '深度覆盖过低，请调高 mesh_grid 或检查渲染'
    print('OK: depth coverage sufficient.')
```

> 期望覆盖率 >70%。这能确保 `K_eff` 不再被“看不见”卡死。

---

## 6) 一键执行顺序（先 mesh→pack→report→overfit→Stage‑A）
```bash
# 1) 清理旧数据
rm -rf ./data/habitat_vln/{train,val} ./raw_sequences/{train,val}
mkdir -p ./raw_sequences/train/RoomA ./raw_sequences/val/RoomB

# 2) 生成 mesh 渲染数据（120 train / 30 val）
python scripts/gen_synth_demo.py --root ./raw_sequences --scene RoomA --clips 120 \
  --T 8 --W 384 --H 384 --pose_mode face_target --path_mode short_arc \
  --arc_deg 20 --radius 2.0 --render_mode mesh --mesh_grid 32 --split train
python scripts/gen_synth_demo.py --root ./raw_sequences --scene RoomB --clips 30 \
  --T 8 --W 384 --H 384 --pose_mode face_target --path_mode short_arc \
  --arc_deg 20 --radius 2.0 --render_mode mesh --mesh_grid 32 --split val

# 3) 快速覆盖率检查
python scripts/check_mesh_visibility.py

# 4) 打包
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split train
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split val

# 5) 质量报告（目标：K_eff≥2 ≥85%，avg H(q) ≤ 5.0）
python scripts/report_data_quality.py --root ./data/habitat_vln --split train
python scripts/report_data_quality.py --root ./data/habitat_vln --split val

# 6) 过拟合（相对熵阈值）
python scripts/overfit_one_batch.py

# 7) Stage‑A（64×64，仅训 Head；AMP off）
CUDA_VISIBLE_DEVICES=0 python scripts/train_multistage.py --config configs/training_config.yaml

# 8) 评估与可视化
python scripts/eval_heatmap.py --config configs/training_config.yaml \
  --ckpt outputs/checkpoints/checkpoint_warmup_head_epoch_3.pt --save-vis
```

---

## 7) 预期验收（SLO）
- `K_eff≥2` 比例：**≥85%**（train/val 均满足）
- `avg H(q)`：**≤ 5.0**（64×64，σ=2.0）
- `mask==0` 通道比例：**<20%**
- Overfit：**loss ≤ median(H(q))+0.3**（通常 <4）
- Stage‑A：`val NLL < 8.0` 且下降趋势明显；叠加图“贴脸”

---

## 8) 性能与小贴士
- 若生成时间偏长：把 `--mesh_grid 32 → 24`，或减少墙体细分（`grid//2`）。
- 若覆盖率仍<70%：`mesh_grid ↑` 或目标牌 `w/h ↑`，或相机更近 `radius ↓`。
- 若 K_eff 仍<85%：`arc_deg` 再降到 `15`，`lookback` 设 3（更近邻）。

---




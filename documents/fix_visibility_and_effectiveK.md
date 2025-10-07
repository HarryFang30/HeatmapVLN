fix_visibility_and_effectiveK.md｜关键帧可见性 & K_eff 修复手册（给 Claude）

目的：解决

No visible keyframes found for ref_idx=7 频繁出现

质量报告里 K_eff 只有 0/1 ⇒ 训练只能学到单张热图

Overfit 判定门槛不科学（H(q) 下界）

修复思路一共 6 步：先修 投影链路 → 自测断言 → 重新 pack & 报告 → 科学化 overfit 门槛 → 再开 Stage-A。

0) 预期改动范围

src/data/heatmap_builder.py（必改）

scripts/pack_dataset.py（必改）

scripts/overfit_one_batch.py（必改）

（可选）src/data/vln_heatmap_adapter.py 小改日志

configs/dataset_pack.yaml（临时调参）

1) 修正：可见性评估使用参考帧内参 Ki（不是 Kj）
1.1 修改函数签名与实现

文件：src/data/heatmap_builder.py
函数：visible_ratio_for_keyframe(...)

把签名里新增 Ki（参考帧内参）参数，并在投影到参考帧时使用 Ki：

# BEFORE (示意)
def visible_ratio_for_keyframe(depth_j, Kj, T_w_c_j, T_c_ref_w, depth_ref, ref_w, ref_h, occl_eps, subsample=4):
    ...
    xi = Kj['fx'] * (pts_ci[:,0]/zi) + Kj['cx']
    yi = Kj['fy'] * (pts_ci[:,1]/zi) + Kj['cy']
    ...

# AFTER
def visible_ratio_for_keyframe(depth_j, Kj, Ki, T_w_c_j, T_c_ref_w,
                               depth_ref, ref_w, ref_h, occl_eps, subsample=4):
    """
    depth_j: 候选关键帧 j 的深度；Kj: 帧 j 的内参；Ki: 参考帧 i 的内参；
    T_w_c_j: 世界←相机j 的4x4齐次矩阵；T_c_ref_w: 参考相机i←世界 的4x4齐次矩阵
    """
    ...
    # 2) 像素→相机j（用 Kj）
    rays = (Kj['Kinv'] @ pix.T).T
    pts_cj = rays * z[:, None]

    # 3) 相机j→世界→参考相机i
    pts_w  = world_from_cam(pts_cj, T_w_c_j)       # Rj * Xcj + tj
    pts_ci = cam_from_world(pts_w, T_c_ref_w)      # Ri * Xw + ti
    zi = pts_ci[:, 2]
    infront = zi > 0
    if infront.sum() == 0:
        return 0.0

    # 4) 参考相机坐标 → 参考像素（用 Ki！！！）
    xi = Ki['fx'] * (pts_ci[:,0]/zi) + Ki['cx']
    yi = Ki['fy'] * (pts_ci[:,1]/zi) + Ki['cy']
    inb = (xi >= 0) & (xi < ref_w) & (yi >= 0) & (yi < ref_h)
    valid = infront & inb
    ...


必须确认：T_c_ref_w = (T_w_c_ref)^{-1}，且矩阵顺序与 world_from_cam / cam_from_world 的约定一致。

2) 同步修正：热力图生成里所有“投到参考帧”的地方也要用 Ki

文件：src/data/heatmap_builder.py（或你实际画热图的模块）

新增一个小工具函数，强制所有投影到参考帧都走这里：

def project_to_ref_pixels(pts_ci, Ki):
    zi = pts_ci[:, 2]
    xi = Ki['fx'] * (pts_ci[:,0]/zi) + Ki['cx']
    yi = Ki['fy'] * (pts_ci[:,1]/zi) + Ki['cy']
    return xi, yi, zi


然后把热力图构建流程里“世界/参考相机坐标 → 像素”的那两行改成调用 project_to_ref_pixels(..., Ki)。

重要：可见性评估与热力图绘制必须一致，否则报告能过、训练图还会飘。

3) 关键帧选择逻辑：传入 Ki & 临时放宽阈值观测“放量”情况

文件：scripts/pack_dataset.py

在调用 visible_ratio_for_keyframe 处，按参考帧 i 构造 Ki 并传入。

临时将 min_visible_ratio 设为 0.0（仅一次），目的是验证修复是否让候选数量“放量”。

仍保留 near-ref fallback（你已实现），但期待它很少触发。

示例（伪码片段）：

Ki = build_intrinsics(W, H, fx=fx_i, fy=fy_i, cx=cx_i, cy=cy_i, return_inv=False)
...
score = visible_ratio_for_keyframe(
    depth_j=depth[j], Kj=Kj, Ki=Ki,
    T_w_c_j=T_w_c[j], T_c_ref_w=np.linalg.inv(T_w_c[ref_idx]),
    depth_ref=None if not cfg['heatmap']['occlusion_check'] else depth[ref_idx],
    ref_w=W, ref_h=H, occl_eps=cfg['heatmap']['occlusion_eps'],
    subsample=cfg['pack'].get('depth_subsample', 4)
)

4) 两个自测断言（必写，秒定位矩阵/内参错位）

新增脚本 scripts/check_visibility_asserts.py：

import os, json, numpy as np
from src.data.heatmap_builder import visible_ratio_for_keyframe, build_intrinsics
# 替换为你项目里世界/相机坐标变换的函数
from src.data.heatmap_builder import world_from_cam, cam_from_world

def run_one_clip(clip_dir):
    meta = json.load(open(os.path.join(clip_dir, 'meta.json')))
    ref = meta.get('ref_idx', 7)
    W, H = meta['W'], meta['H']
    T_w_c = [np.array(M, dtype=np.float32) for M in meta['T_w_c']]
    # 自定义的深度加载
    depth = np.load(os.path.join(clip_dir, 'depth.npy'))  # [T,H,W] 或按你实际存法

    Ki = build_intrinsics(W, H, fx=meta['ref_intrinsics']['fx'], fy=meta['ref_intrinsics']['fy'],
                          cx=meta['ref_intrinsics']['cx'], cy=meta['ref_intrinsics']['cy'])
    Kj_ref = build_intrinsics(W, H, fx=meta['k_intrinsics'][ref]['fx'], fy=meta['k_intrinsics'][ref]['fy'],
                              cx=meta['k_intrinsics'][ref]['cx'], cy=meta['k_intrinsics'][ref]['cy'])

    # 1) j=ref → 应 ≈ 1.0 （关闭 occlusion 时）
    vis_ref = visible_ratio_for_keyframe(depth[ref], Kj_ref, Ki, T_w_c[ref],
                                         np.linalg.inv(T_w_c[ref]),
                                         depth_ref=None, ref_w=W, ref_h=H,
                                         occl_eps=0.0, subsample=4)
    print('vis(ref→ref)=', vis_ref)
    assert vis_ref >= 0.95, 'vis(ref→ref) 应接近 1.0 —— 检查 T 或 Ki/Kj 使用'

    # 2) j=ref-1 → 应显著 > 0
    j = max(0, ref-1)
    Kj_prev = build_intrinsics(W, H, fx=meta['k_intrinsics'][j]['fx'], fy=meta['k_intrinsics'][j]['fy'],
                               cx=meta['k_intrinsics'][j]['cx'], cy=meta['k_intrinsics'][j]['cy'])
    vis_prev = visible_ratio_for_keyframe(depth[j], Kj_prev, Ki, T_w_c[j],
                                          np.linalg.inv(T_w_c[ref]),
                                          depth_ref=None, ref_w=W, ref_h=H,
                                          occl_eps=0.0, subsample=4)
    print('vis(prev→ref)=', vis_prev)
    assert vis_prev > 0.1, 'vis(prev→ref) 应该 > 0.1 —— 检查弧度/朝向/投影公式'

if __name__ == '__main__':
    # 随便挑一个生成好的 clip 跑断言
    run_one_clip('./raw_sequences/train/RoomA/clip_000001')

5) Overfit 判定：从“绝对值 <1”改为“相对熵阈值”

文件：scripts/overfit_one_batch.py

在选择子集（1–2 个样本）后，计算目标分布的中位熵，并以 阈值 = median(H(q)) + 0.3 作为 PASS 线：

from src.data.quality_metrics import heatmap_entropy
import torch

# after building subset `small` and DataLoader -> 拿一批 GT
batch = next(iter(dl))
H = heatmap_entropy(batch['gt_heatmaps'])  # [K] or [B,K]
H_med = float(torch.median(H[batch['mask'].bool()])) if batch['mask'].sum() > 0 else 4.0
pass_threshold = H_med + 0.3
...
# 训练结束后:
print(f"Final loss: {loss:.4f} | PASS threshold: {pass_threshold:.4f}")
print("PASS" if float(loss) <= pass_threshold else "FAIL")


解释：交叉熵的最优值就是 H(q)；不可能一律 <1。+0.3 给点余量，避免小样本噪声。

6) 一次性执行顺序（用于这轮修复验证）

临时调参（验证修复是否“放量”）：

configs/dataset_pack.yaml：

pack.min_visible_ratio: 0.0

heatmap.occlusion_check: false

其它保持不动（gaussian_sigma_px: 1.8 可保留）

命令：

# A. 运行断言（先挑 1 个 clip 自测）
python scripts/check_visibility_asserts.py

# B. 重打包
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split train
python scripts/pack_dataset.py --config configs/dataset_pack.yaml --split val

# C. 质量报告（期望 K_eff≥2 明显上升到 ≥70%）
python scripts/report_data_quality.py --root ./data/habitat_vln --split train
python scripts/report_data_quality.py --root ./data/habitat_vln --split val

# D. 过拟合（以 H(q)+0.3 为门槛，期望 PASS）
python scripts/overfit_one_batch.py

# E. Stage-A（64×64，仅训 Head；AMP off）
CUDA_VISIBLE_DEVICES=0 python scripts/train_multistage.py --config configs/training_config.yaml

# F. 评估与可视化
python scripts/eval_heatmap.py --config configs/training_config.yaml \
  --ckpt outputs/checkpoints/checkpoint_warmup_head_epoch_3.pt --save-vis


若 B/C 仍显示 No visible keyframes 或 K_eff<2 比例高：

再查 scripts/check_visibility_asserts.py 的两条断言输出：

vis(ref→ref) < 0.95 ⇒ 强烈指向 T 矩阵方向或 world_from_cam/cam_from_world 的实现问题；

vis(prev→ref) ≈ 0 但 ref→ref 正常 ⇒ 检查 Ki/Kj 传参、ref_idx 是否为 T-1、短弧/朝向是否生效。

生成器参数可再加强重叠：arc_deg: 20（更短弧）、radius: 2.0、noise_rot_deg: 1。

修复验证通过后，把 pack.min_visible_ratio 恢复到 0.01 或 0.02，再跑一次 pack+report，目标 K_eff≥2 ≥80%。
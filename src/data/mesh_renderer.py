# src/data/mesh_renderer.py
import numpy as np

# ---------------------- 数学与相机 ----------------------

def invert_se3(T):
    """反转SE(3)变换矩阵"""
    R = T[:3,:3]
    t = T[:3,3:4]
    Rt = R.T
    Tinv = np.eye(4, dtype=np.float32)
    Tinv[:3,:3] = Rt
    Tinv[:3,3:4] = -Rt @ t
    return Tinv

def cam_from_world(Xw, T_c_w):
    """将世界坐标点转换到相机坐标系
    Args:
        Xw: [N,3] 世界坐标点
        T_c_w: 4x4 相机到世界的变换矩阵
    Returns:
        [N,3] 相机坐标系中的点
    """
    N = Xw.shape[0]
    Xh = np.concatenate([Xw, np.ones((N,1), dtype=np.float32)], axis=1)  # [N,4]
    Xc = (T_c_w @ Xh.T).T  # [N,4]
    return Xc[:, :3]

def project_pixels(Xc, K):
    """将相机坐标系点投影到像素坐标
    Args:
        Xc: [N,3] 相机坐标系点
        K: dict with fx,fy,cx,cy
    Returns:
        x, y, z: 像素坐标和深度
    """
    z = Xc[:,2]
    x = K['fx'] * (Xc[:,0] / z) + K['cx']
    y = K['fy'] * (Xc[:,1] / z) + K['cy']
    return x, y, z

# ---------------------- 栅格化核心 ----------------------

def _bbox2d(xs, ys, W, H):
    """计算2D包围盒"""
    xmin = max(int(np.floor(xs.min())), 0)
    xmax = min(int(np.ceil(xs.max())), W-1)
    ymin = max(int(np.floor(ys.min())), 0)
    ymax = min(int(np.ceil(ys.max())), H-1)
    return xmin, xmax, ymin, ymax

def _barycentric(px, py, x0, y0, x1, y1, x2, y2):
    """计算像素中心相对三角形的重心坐标"""
    den = (y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2)
    w0 = ((y1 - y2)*(px - x2) + (x2 - x1)*(py - y2)) / (den + 1e-12)
    w1 = ((y2 - y0)*(px - x2) + (x0 - x2)*(py - y2)) / (den + 1e-12)
    w2 = 1.0 - w0 - w1
    return w0, w1, w2

def rasterize_triangles(verts_w, faces, colors_v, K, T_c_w, W, H,
                        z_far=100.0, z_near=0.05, backface_cull=False):
    """
    三角形栅格化渲染器（优化版本）
    Args:
        verts_w: [V,3] 世界坐标顶点
        faces: [F,3] 顶点索引
        colors_v: [V,3] 顶点颜色(0..1)
        K: dict with fx, fy, cx, cy (内参)
        T_c_w: 4x4 相机到世界的变换
        W, H: 图像宽高
        z_far, z_near: 远近裁剪平面
        backface_cull: 是否背面剔除
    Returns:
        rgb: [H,W,3] RGB图像
        depth: [H,W] 深度图(0表示无效深度)
    """
    # 1) 世界->相机->像素
    verts_c = cam_from_world(verts_w, T_c_w)

    rgb = np.zeros((H, W, 3), dtype=np.float32)
    depth = np.zeros((H, W), dtype=np.float32)
    zbuf = np.full((H, W), np.inf, dtype=np.float32)

    # 批量投影所有顶点
    fx, fy, cx, cy = K['fx'], K['fy'], K['cx'], K['cy']
    x_proj = fx * (verts_c[:, 0] / (verts_c[:, 2] + 1e-8)) + cx
    y_proj = fy * (verts_c[:, 1] / (verts_c[:, 2] + 1e-8)) + cy
    z_proj = verts_c[:, 2]

    for f in faces:
        v0, v1, v2 = f

        # 获取投影后的顶点坐标
        x0, y0, z0 = x_proj[v0], y_proj[v0], z_proj[v0]
        x1, y1, z1 = x_proj[v1], y_proj[v1], z_proj[v1]
        x2, y2, z2 = x_proj[v2], y_proj[v2], z_proj[v2]

        # 全部在近裁剪平面之前则跳过
        if z0 <= z_near and z1 <= z_near and z2 <= z_near:
            continue

        # 背面剔除
        if backface_cull:
            area = (x1-x0)*(y2-y0) - (y1-y0)*(x2-x0)
            if area >= 0:
                continue

        # 2D 包围盒
        xmin, xmax, ymin, ymax = _bbox2d(
            np.array([x0, x1, x2]),
            np.array([y0, y1, y2]),
            W, H
        )
        if xmin > xmax or ymin > ymax:
            continue

        # 顶点颜色
        c0 = colors_v[v0]
        c1 = colors_v[v1]
        c2 = colors_v[v2]

        # 向量化计算重心坐标（批处理一行）
        for yy in range(ymin, ymax+1):
            py = yy + 0.5
            px = np.arange(xmin, xmax+1) + 0.5

            # 批量计算重心坐标
            den = (y1 - y2)*(x0 - x2) + (x2 - x1)*(y0 - y2) + 1e-12
            w0 = ((y1 - y2)*(px - x2) + (x2 - x1)*(py - y2)) / den
            w1 = ((y2 - y0)*(px - x2) + (x0 - x2)*(py - y2)) / den
            w2 = 1.0 - w0 - w1

            # 找到在三角形内的像素
            inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)

            if not np.any(inside):
                continue

            # 计算深度
            z = w0*z0 + w1*z1 + w2*z2

            # 深度测试和更新
            xx = np.arange(xmin, xmax+1)[inside]
            z_valid = z[inside]
            w0_valid = w0[inside]
            w1_valid = w1[inside]
            w2_valid = w2[inside]

            for i, x_i in enumerate(xx):
                z_i = z_valid[i]
                if z_near < z_i < z_far and z_i < zbuf[yy, x_i]:
                    zbuf[yy, x_i] = z_i
                    depth[yy, x_i] = z_i
                    rgb[yy, x_i, :] = w0_valid[i]*c0 + w1_valid[i]*c1 + w2_valid[i]*c2

    return rgb, depth

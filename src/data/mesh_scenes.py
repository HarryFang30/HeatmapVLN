# src/data/mesh_scenes.py
import numpy as np

# 默认颜色定义
COL_FLOOR = np.array([0.7, 0.7, 0.7], dtype=np.float32)
COL_WALL  = np.array([0.8, 0.85, 0.9], dtype=np.float32)
COL_OBJ   = np.array([0.9, 0.2, 0.2], dtype=np.float32)


def _grid_plane(xmin, xmax, zmin, zmax, y, nx=32, nz=32, color=(1, 1, 1)):
    """生成水平面 (y 常数) 的网格细分（地板）
    Args:
        xmin, xmax, zmin, zmax: 平面边界
        y: y坐标（高度）
        nx, nz: x和z方向的细分数
        color: 顶点颜色
    Returns:
        verts: [V,3] 顶点坐标
        faces: [F,3] 三角形索引
        colors: [V,3] 顶点颜色
    """
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

    def vid(i, k):
        return i*(nz+1) + k

    for i in range(nx):
        for k in range(nz):
            v00 = vid(i, k)
            v10 = vid(i+1, k)
            v01 = vid(i, k+1)
            v11 = vid(i+1, k+1)
            # 两个三角形组成一个四边形
            faces.append([v00, v10, v11])
            faces.append([v00, v11, v01])

    faces = np.array(faces, dtype=np.int32)
    return verts, faces, colors


def _grid_wall(z, xmin, xmax, ymin, ymax, nx=16, ny=16, color=(1, 1, 1)):
    """生成竖直墙（z 常数）
    Args:
        z: z坐标（墙的位置）
        xmin, xmax, ymin, ymax: 墙的边界
        nx, ny: x和y方向的细分数
        color: 顶点颜色
    Returns:
        verts: [V,3] 顶点坐标
        faces: [F,3] 三角形索引
        colors: [V,3] 顶点颜色
    """
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

    def vid(i, j):
        return i*(ny+1) + j

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


def _billboard(center=(0, 1.2, 2.0), w=0.6, h=0.6, nx=8, ny=8, color=(0.9, 0.2, 0.2)):
    """生成广告牌/目标物体（垂直平面）
    Args:
        center: 中心位置 (x, y, z)
        w, h: 宽度和高度
        nx, ny: x和y方向的细分数
        color: 顶点颜色
    Returns:
        verts: [V,3] 顶点坐标
        faces: [F,3] 三角形索引
        colors: [V,3] 顶点颜色
    """
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

    def vid(i, j):
        return i*(ny+1) + j

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
    """构建简单室内场景：地板 + 两面墙 + 目标广告牌
    Args:
        grid: 网格细分密度（越大覆盖率越高）
    Returns:
        verts: [V,3] 所有顶点
        faces: [F,3] 所有三角形索引
        colors: [V,3] 所有顶点颜色
    """
    # 地板：x∈[-2,2], z∈[0.5,3.5], y=0
    v0, f0, c0 = _grid_plane(
        -2, 2, 0.5, 3.5,
        y=0.0,
        nx=grid, nz=grid,
        color=COL_FLOOR
    )

    # 后墙：z=3.5，x∈[-2,2], y∈[0,2.5]
    v1, f1, c1 = _grid_wall(
        3.5, -2, 2, 0.0, 2.5,
        nx=grid//2, ny=grid//2,
        color=COL_WALL
    )

    # 前墙：z=0.5，x∈[-2,2], y∈[0,2.5]
    v2, f2, c2 = _grid_wall(
        0.5, -2, 2, 0.0, 2.5,
        nx=grid//2, ny=grid//2,
        color=COL_WALL
    )

    # 目标广告牌：中心(0,1.2,2.0)，细分提升覆盖
    v3, f3, c3 = _billboard(
        center=(0, 1.2, 2.0),
        w=0.8, h=0.8,
        nx=grid//2, ny=grid//2,
        color=COL_OBJ
    )

    # 拼接所有几何体
    verts = np.concatenate([v0, v1, v2, v3], axis=0)
    colors = np.concatenate([c0, c1, c2, c3], axis=0)

    # 调整索引偏移
    off1 = len(v0)
    off2 = len(v0) + len(v1)
    off3 = len(v0) + len(v1) + len(v2)

    faces = np.concatenate([
        f0,
        f1 + off1,
        f2 + off2,
        f3 + off3
    ], axis=0)

    return verts, faces, colors

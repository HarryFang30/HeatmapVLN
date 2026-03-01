#!/usr/bin/env python3
"""
轨迹热力图可视化
================

沿着一条完整轨迹，可视化每个采样位置的：
- RGB 帧 + GT 热力图叠加
- RGB 帧 + 预测热力图叠加

输出一张大的 grid 图，展示轨迹上热力图的时序变化。

用法:
    python scripts/visualize_trajectory_heatmaps.py \
        --checkpoint /path/to/best.pth \
        --num-clips 3 \
        --frames-per-clip 12 \
        --output-dir ./vis_trajectory \
        --inference-steps 200
"""

import os
import sys
import json
import random
import argparse
import logging
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import yaml
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib
matplotlib.use('Agg')
import cv2

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.models.pipeline import VLNPipeline, VLNPipelineConfig
from src.utils.gpu_heatmap import GPUHeatmapComputer

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("vis_traj")


# ==================== build_model (与 train.py 一致) ====================
def build_model(cfg: Dict) -> VLNPipeline:
    model_cfg = cfg['model']
    llm_cfg = model_cfg.get('llm', {})
    heatmap_cfg = model_cfg.get('heatmap_head', {})
    action_cfg = model_cfg.get('action_head', {})
    progress_cfg = model_cfg.get('progress_head', {})
    action_head_type = action_cfg.get('type', 'transformer')

    config = VLNPipelineConfig(
        llm_model_path=llm_cfg.get('model_path', './models/qwen_3_vl'),
        llm_hidden_dim=llm_cfg.get('hidden_dim', 2048),
        llm_token_dim=llm_cfg.get('token_dim', 1024),
        llm_torch_dtype=llm_cfg.get('torch_dtype', 'bfloat16'),
        llm_attn_implementation=llm_cfg.get('attn_implementation', 'flash_attention_2'),
        max_video_frames=llm_cfg.get('max_video_frames', 16),
        enable_packing=False,
        max_seq_length=llm_cfg.get('max_seq_length', 4096),
        spatial_merge_size=llm_cfg.get('spatial_merge_size', 2),
        device=model_cfg.get('device', 'cuda'),
        heatmap_size=tuple(cfg['data']['init_hm_size']),
        enable_history_heatmap_head=heatmap_cfg.get('enable_history', True),
        enable_future_heatmap_head=heatmap_cfg.get('enable_future', False),
        diffusion_heatmap_cond_dim=heatmap_cfg.get('cond_dim', 512),
        diffusion_heatmap_num_inference_steps=heatmap_cfg.get('num_inference_steps', 10),
        image_size=cfg['data']['image_size'][0],
        heatmap_use_image_encoder=heatmap_cfg.get('use_image_encoder', True),
        heatmap_pool_method=heatmap_cfg.get('pool_method', 'attention'),
        heatmap_pool_num_heads=heatmap_cfg.get('pool_num_heads', 4),
        heatmap_use_circular_padding=heatmap_cfg.get('use_circular_padding', False),
        heatmap_dropout=heatmap_cfg.get('dropout', 0.1),
        heatmap_block_out_channels=tuple(heatmap_cfg.get('block_out_channels', [64, 128, 256])),
        heatmap_layers_per_block=heatmap_cfg.get('layers_per_block', 2),
        heatmap_attention_levels=tuple(heatmap_cfg.get('attention_levels', [2])),
        heatmap_num_train_timesteps=heatmap_cfg.get('num_train_timesteps', 100),
        heatmap_cfg_drop_prob=heatmap_cfg.get('cfg_drop_prob', 0.1),
        heatmap_cfg_scale=heatmap_cfg.get('cfg_scale', 3.0),
        heatmap_use_sequence_conditioning=heatmap_cfg.get('use_sequence_conditioning', False),
        heatmap_seq_cross_attn_heads=heatmap_cfg.get('seq_cross_attn_heads', 8),
        heatmap_seq_cross_attn_head_dim=heatmap_cfg.get('seq_cross_attn_head_dim', 64),
        heatmap_use_spatial_injection=heatmap_cfg.get('use_spatial_injection', False),
        heatmap_image_encoder_use_pretrained=heatmap_cfg.get('image_encoder_use_pretrained', False),
        use_lora=llm_cfg.get('use_lora', False),
        lora_rank=llm_cfg.get('lora_rank', 16),
        lora_alpha=llm_cfg.get('lora_alpha', 32),
        lora_num_layers=llm_cfg.get('lora_num_layers', 4),
        lora_dropout=llm_cfg.get('lora_dropout', 0.05),
        lora_target_modules=llm_cfg.get('lora_target_modules', None),
        action_head_type=action_head_type,
        enable_action_head=False,
        enable_stop_head=False,
        enable_progress_head=False,
        verbose=False,
    )
    return VLNPipeline(config)


def find_good_clips(data_root: str, min_frames: int = 60, num_clips: int = 3, seed: int = 42):
    """找到足够长的 clips 以展示轨迹"""
    data_root = Path(data_root)
    good_clips = []
    for scene_dir in sorted(data_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        for clip_dir in sorted(scene_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            rgb_dir = clip_dir / "rgb"
            poses_file = clip_dir / "poses.json"
            if not rgb_dir.exists() or not poses_file.exists():
                continue
            n_frames = len(list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png")))
            if n_frames >= min_frames:
                good_clips.append((clip_dir, n_frames))

    random.seed(seed)
    random.shuffle(good_clips)
    return good_clips[:num_clips]


def load_clip_data(clip_dir: Path, image_size: Tuple[int, int] = (224, 224)):
    """加载一个 clip 的全部数据"""
    # Frames
    rgb_dir = clip_dir / "rgb"
    frame_files = sorted(list(rgb_dir.glob("*.jpg")) + list(rgb_dir.glob("*.png")))

    # Poses
    with open(clip_dir / "poses.json") as f:
        all_poses = json.load(f)

    # Intrinsics
    intrinsics = None
    intrinsics_file = clip_dir / "intrinsics.json"
    if intrinsics_file.exists():
        with open(intrinsics_file) as f:
            intr = json.load(f)
        if "K" in intr:
            intrinsics = np.array(intr["K"], dtype=np.float32)

    # Instruction
    instruction = "Navigate to the destination."
    meta_file = clip_dir / "meta.json"
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        instruction = meta.get("instruction", instruction)

    n_frames = min(len(frame_files), len(all_poses))

    # Load all frames as tensors
    frames_np = []  # for visualization (HWC, 0-1)
    frames_tensor = []  # for model (CHW, 0-1)
    for i in range(n_frames):
        img = cv2.imread(str(frame_files[i]))
        if img is None:
            continue
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, image_size)
        frames_np.append(img_resized.astype(np.float32) / 255.0)
        t = torch.from_numpy(img_resized).float() / 255.0
        frames_tensor.append(t.permute(2, 0, 1))  # HWC -> CHW

    return {
        "clip_dir": clip_dir,
        "frames_np": frames_np,
        "frames_tensor": frames_tensor,
        "poses": [np.array(p, dtype=np.float32) for p in all_poses[:len(frames_np)]],
        "intrinsics": intrinsics,
        "instruction": instruction,
        "n_frames": len(frames_np),
    }


def overlay_heatmap_on_frame(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """将热力图叠加到 RGB 帧上 (frame: HWC 0-1, heatmap: HxW 0-1)"""
    H, W = frame.shape[:2]
    hm_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_CUBIC)

    # 使用 inferno colormap
    hm_uint8 = (np.clip(hm_resized, 0, 1) * 255).astype(np.uint8)
    hm_colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_INFERNO)
    hm_colored = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

    # Selective overlay: 只在热力图值 > threshold 的区域叠加
    threshold = 0.05
    mask = hm_resized > threshold
    mask_3d = np.stack([mask, mask, mask], axis=-1).astype(np.float32)

    blended = frame * (1 - alpha) + hm_colored * alpha
    result = np.where(mask_3d > 0, blended, frame)
    return np.clip(result, 0, 1)


def render_birdseye_view(
    all_poses: List[np.ndarray],
    sample_positions: List[int],
    peak_dists: List[float],
    view_w: int,
    view_h: int,
    margin: int = 30,
) -> np.ndarray:
    """
    渲染俯视图（Bird's-Eye View）：展示 agent 在世界坐标系中的完整轨迹。

    - 灰色细线：完整轨迹
    - 彩色圆点：采样位置（绿=<5px，黄=<15px，红=>=15px）
    - 箭头：agent 朝向（相机 -Z 方向在 XZ 平面的投影）
    - 数字标注：帧号
    - 绿色星号：起点，红色方块：终点
    """
    # 提取所有位置 (X, Z 平面，Y 是 up)
    positions = []
    for p in all_poses:
        pos = np.array(p, dtype=np.float32)
        # 相机位置 = pose[:3, 3]（world-from-camera 矩阵的平移列）
        x, y, z = pos[0, 3], pos[1, 3], pos[2, 3]
        positions.append((x, z))  # 使用 X-Z 平面作为俯视图

    positions = np.array(positions)  # [N, 2]

    # 计算坐标范围并归一化到画布
    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    z_min, z_max = positions[:, 1].min(), positions[:, 1].max()

    # 增加边距
    x_range = max(x_max - x_min, 0.1)
    z_range = max(z_max - z_min, 0.1)
    # 保持宽高比
    scale = min((view_w - 2 * margin) / x_range, (view_h - 2 * margin) / z_range)

    def world_to_px(wx, wz):
        px = int(margin + (wx - x_min) * scale)
        py = int(margin + (wz - z_min) * scale)
        # 翻转 Y 使 Z+ 朝上
        py = view_h - py
        return px, py

    # 绘制画布（白底）
    canvas = np.ones((view_h, view_w, 3), dtype=np.uint8) * 245

    # 1. 绘制完整轨迹（灰色细线）
    pts = [world_to_px(x, z) for x, z in positions]
    for i in range(len(pts) - 1):
        cv2.line(canvas, pts[i], pts[i + 1], (180, 180, 180), 1, cv2.LINE_AA)

    # 2. 起点（绿色星号）和终点（红色方块）
    cv2.drawMarker(canvas, pts[0], (0, 180, 0), cv2.MARKER_STAR, 16, 2)
    cv2.drawMarker(canvas, pts[-1], (0, 0, 200), cv2.MARKER_SQUARE, 10, 2)

    # 3. 绘制采样位置
    for idx, (pos_frame, dist) in enumerate(zip(sample_positions, peak_dists)):
        if pos_frame >= len(positions):
            continue
        px, py = world_to_px(positions[pos_frame, 0], positions[pos_frame, 1])

        # 颜色：绿 <5px，黄 <15px，红 >=15px
        if dist < 5:
            color = (0, 200, 0)  # 绿
        elif dist < 15:
            color = (0, 200, 255)  # 黄
        elif dist == float('inf'):
            color = (200, 200, 200)  # 灰
        else:
            color = (0, 0, 220)  # 红

        # 圆圈
        cv2.circle(canvas, (px, py), 7, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 7, (0, 0, 0), 1, cv2.LINE_AA)

        # 绘制朝向箭头（相机的 -Z 方向在 XZ 平面的投影）
        pose = np.array(all_poses[pos_frame], dtype=np.float32)
        # 相机前方方向 = -R[:, 2]（负 Z 轴）
        fwd_world = -pose[:3, 2]
        fwd_xz = np.array([fwd_world[0], fwd_world[2]])
        fwd_len = np.linalg.norm(fwd_xz)
        if fwd_len > 1e-6:
            fwd_xz = fwd_xz / fwd_len
            arrow_len = 18
            dx = int(fwd_xz[0] * arrow_len)
            dy = int(-fwd_xz[1] * arrow_len)  # 翻转 Y
            cv2.arrowedLine(canvas, (px, py), (px + dx, py + dy),
                           color, 2, cv2.LINE_AA, tipLength=0.35)

        # 帧号标注
        label = f"F{pos_frame}"
        cv2.putText(canvas, label, (px + 10, py - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (60, 60, 60), 1, cv2.LINE_AA)

    # 4. 图例
    legend_x = 5
    legend_y = view_h - 60
    cv2.putText(canvas, "Bird's-Eye View", (legend_x, 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
    cv2.circle(canvas, (legend_x + 8, legend_y), 5, (0, 200, 0), -1)
    cv2.putText(canvas, "<5px", (legend_x + 18, legend_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    cv2.circle(canvas, (legend_x + 8, legend_y + 16), 5, (0, 200, 255), -1)
    cv2.putText(canvas, "<15px", (legend_x + 18, legend_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    cv2.circle(canvas, (legend_x + 8, legend_y + 32), 5, (0, 0, 220), -1)
    cv2.putText(canvas, ">=15px", (legend_x + 18, legend_y + 36),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    # 起终点图例
    cv2.drawMarker(canvas, (legend_x + 70, legend_y), (0, 180, 0), cv2.MARKER_STAR, 10, 1)
    cv2.putText(canvas, "Start", (legend_x + 80, legend_y + 4),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)
    cv2.drawMarker(canvas, (legend_x + 70, legend_y + 16), (0, 0, 200), cv2.MARKER_SQUARE, 8, 1)
    cv2.putText(canvas, "End", (legend_x + 80, legend_y + 20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.3, (0, 0, 0), 1)

    return canvas


def make_trajectory_grid(
    clip_name: str,
    instruction: str,
    sample_positions: List[int],
    frames_np: List[np.ndarray],
    gt_heatmaps: List[np.ndarray],
    pred_heatmaps: List[np.ndarray],
    peak_dists: List[float],
    all_poses: List[np.ndarray],
    output_path: str,
):
    """
    生成轨迹 grid 可视化:
      左侧: 俯视图（Bird's-Eye View）
      Row 0: 原始 RGB 帧序列
      Row 1: RGB + GT 热力图叠加
      Row 2: RGB + 预测热力图叠加
    """
    N = len(sample_positions)
    H, W = frames_np[0].shape[:2]

    # 3 行 N 列的帧网格
    n_rows = 3
    row_labels = ["RGB Frame", "GT Heatmap", "Pred Heatmap"]

    left_margin = 120  # 行标签
    top_margin = 60    # 标题行
    grid_h = n_rows * H
    grid_w = N * W

    # 俯视图尺寸（与帧网格等高）
    bev_w = max(300, int(grid_h * 0.8))
    bev_h = grid_h

    # 总画布
    canvas_w = left_margin + grid_w + 20 + bev_w + 10
    canvas_h = top_margin + grid_h + 10
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    # ==================== 标题 ====================
    title = f"{clip_name} | {instruction[:90]}{'...' if len(instruction) > 90 else ''}"
    cv2.putText(canvas, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    # ==================== 行标签 ====================
    for row_idx in range(n_rows):
        y_start = top_margin + row_idx * H
        label_y = y_start + H // 2
        cv2.putText(canvas, row_labels[row_idx], (5, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    # ==================== 帧网格 ====================
    for col_idx in range(N):
        pos = sample_positions[col_idx]
        frame = frames_np[col_idx]
        gt_hm = gt_heatmaps[col_idx]
        pred_hm = pred_heatmaps[col_idx]
        dist = peak_dists[col_idx]

        x_start = left_margin + col_idx * W

        for row_idx in range(n_rows):
            y_start = top_margin + row_idx * H

            if row_idx == 0:
                img = (frame * 255).astype(np.uint8)
                img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            elif row_idx == 1:
                overlay = overlay_heatmap_on_frame(frame, gt_hm, alpha=0.6)
                img_bgr = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)
            else:
                overlay = overlay_heatmap_on_frame(frame, pred_hm, alpha=0.6)
                img_bgr = cv2.cvtColor((overlay * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

            # 帧号
            cv2.putText(img_bgr, f"F{pos}", (3, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            # 预测行加 peak_distance
            if row_idx == 2:
                dist_str = f"{dist:.1f}px" if dist != float('inf') else "N/A"
                color = (0, 255, 0) if dist < 5 else (0, 255, 255) if dist < 15 else (0, 0, 255)
                cv2.putText(img_bgr, dist_str, (3, H - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            canvas[y_start:y_start + H, x_start:x_start + W] = img_bgr

    # ==================== 俯视图 ====================
    bev_x_start = left_margin + grid_w + 20
    bev_y_start = top_margin
    bev_img = render_birdseye_view(
        all_poses=all_poses,
        sample_positions=sample_positions,
        peak_dists=peak_dists,
        view_w=bev_w,
        view_h=bev_h,
    )
    canvas[bev_y_start:bev_y_start + bev_h, bev_x_start:bev_x_start + bev_w] = bev_img

    # 俯视图边框
    cv2.rectangle(canvas, (bev_x_start, bev_y_start),
                  (bev_x_start + bev_w - 1, bev_y_start + bev_h - 1),
                  (150, 150, 150), 1)

    cv2.imwrite(output_path, canvas)
    logger.info(f"  Saved: {output_path}")


def main():
    parser = argparse.ArgumentParser(description="轨迹热力图可视化")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num-clips', type=int, default=3, help='可视化的 clip 数')
    parser.add_argument('--frames-per-clip', type=int, default=12, help='每个 clip 采样的帧数')
    parser.add_argument('--output-dir', type=str, default='./vis_trajectory')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--inference-steps', type=int, default=None)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==================== 加载 checkpoint ====================
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    epoch = ckpt.get('epoch', '?')
    logger.info(f"  Epoch: {epoch}")

    if args.inference_steps is not None:
        cfg['model']['heatmap_head']['num_inference_steps'] = args.inference_steps
        logger.info(f"  Override inference_steps = {args.inference_steps}")

    data_root = cfg['data']['root']
    image_size = tuple(cfg['data']['image_size'])
    hm_size = tuple(cfg['data']['init_hm_size'])
    num_history = cfg['data'].get('sliding_window', {}).get('num_history_sample', 32)
    stride = cfg['data'].get('sliding_window', {}).get('sample_stride', 2)

    # ==================== 构建模型 ====================
    logger.info("Building model...")
    model = build_model(cfg)
    state_dict = ckpt.get('trainable_state_dict', {})
    if state_dict:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"  Loaded: {len(state_dict)} params, missing={len(missing)}")
    model = model.to(device)
    model.eval()

    # GPU heatmap computer
    heatmap_computer = GPUHeatmapComputer(hm_size=hm_size, img_size=(640, 480), device=device)

    # ==================== 找 clips ====================
    min_frames_needed = num_history * stride + args.frames_per_clip * stride + 10
    logger.info(f"Finding clips with >= {min_frames_needed} frames...")
    clips = find_good_clips(data_root, min_frames=min_frames_needed,
                            num_clips=args.num_clips, seed=args.seed)
    logger.info(f"  Found {len(clips)} clips")

    # ==================== 处理每个 clip ====================
    for clip_idx, (clip_dir, n_total) in enumerate(clips):
        clip_name = f"{clip_dir.parent.name}/{clip_dir.name}"
        logger.info(f"\n[Clip {clip_idx+1}/{len(clips)}] {clip_name} ({n_total} frames)")

        data = load_clip_data(clip_dir, image_size=image_size)
        n = data["n_frames"]
        logger.info(f"  Loaded {n} frames, instruction: {data['instruction'][:60]}...")

        # 选择采样位置（沿轨迹均匀采样）
        # 每个位置需要至少 num_history*stride 帧的历史
        min_pos = num_history * stride
        max_pos = n - 1
        if max_pos <= min_pos:
            logger.warning(f"  Clip too short, skipping")
            continue

        sample_positions = np.linspace(min_pos, max_pos, args.frames_per_clip, dtype=int).tolist()
        # 去重
        sample_positions = sorted(set(sample_positions))

        logger.info(f"  Sampling {len(sample_positions)} positions: {sample_positions[:5]}...{sample_positions[-3:]}")

        frames_for_vis = []
        gt_heatmaps = []
        pred_heatmaps = []
        peak_dists = []

        model_loaded = False

        for pos_idx, pos in enumerate(sample_positions):
            # 构建历史帧索引
            history_end = pos
            history_indices = []
            idx = history_end - stride
            while idx >= 0 and len(history_indices) < num_history:
                history_indices.append(idx)
                idx -= stride
            history_indices = sorted(history_indices)
            while len(history_indices) < num_history:
                history_indices.append(history_indices[-1] if history_indices else 0)

            current_idx = pos

            # 构建 video_frames = history + current
            all_indices = history_indices + [current_idx]
            video_frames = torch.stack([data["frames_tensor"][i] for i in all_indices])  # [T, C, H, W]
            video_frames = video_frames.unsqueeze(0).to(device)  # [1, T, C, H, W]
            current_obs = video_frames[:, -1]  # [1, C, H, W]

            # GT 热力图
            history_poses = torch.tensor(
                np.stack([data["poses"][i] for i in history_indices]),
                dtype=torch.float32
            ).unsqueeze(0).to(device)  # [1, K, 4, 4]
            current_pose = torch.tensor(
                data["poses"][current_idx], dtype=torch.float32
            ).unsqueeze(0).to(device)  # [1, 4, 4]

            gt_K = None
            if data["intrinsics"] is not None:
                gt_K = torch.tensor(data["intrinsics"], dtype=torch.float32).unsqueeze(0).to(device)

            with torch.no_grad():
                gt_hm = heatmap_computer.compute_batch(
                    history_poses=history_poses,
                    current_poses=current_pose,
                    current_depths=None,
                    intrinsics=gt_K,
                )[0].cpu().numpy()  # [Hm, Wm]

            # 模型推理
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                vis_output = model(
                    video_frames=video_frames,
                    instruction_text=[data["instruction"]],
                    current_observation=current_obs,
                    return_heatmaps=True,
                    return_actions=False,
                )

            if 'history_heatmaps' in vis_output and vis_output['history_heatmaps'] is not None:
                pred_hm = vis_output['history_heatmaps'][0, -1].float().cpu().numpy()
                pred_hm = np.clip(pred_hm, 0, 1)
            else:
                pred_hm = np.zeros(hm_size, dtype=np.float32)

            # 计算 peak distance
            if gt_hm.max() > 0.01 and pred_hm.max() > 0.01:
                gt_peak = np.unravel_index(gt_hm.argmax(), gt_hm.shape)
                pred_peak = np.unravel_index(pred_hm.argmax(), pred_hm.shape)
                dist = np.sqrt((gt_peak[0] - pred_peak[0])**2 + (gt_peak[1] - pred_peak[1])**2)
            else:
                dist = float('inf')

            frames_for_vis.append(data["frames_np"][current_idx])
            gt_heatmaps.append(gt_hm)
            pred_heatmaps.append(pred_hm)
            peak_dists.append(dist)

            logger.info(f"    Pos {pos:>4d}: peak_dist={dist:>5.1f}px, "
                         f"pred_max={pred_hm.max():.3f}, gt_max={gt_hm.max():.3f}")

        # 生成 grid 图（含俯视图）
        out_path = output_dir / f"clip_{clip_idx:02d}_{clip_dir.name}.png"
        make_trajectory_grid(
            clip_name=clip_name,
            instruction=data["instruction"],
            sample_positions=sample_positions,
            frames_np=frames_for_vis,
            gt_heatmaps=gt_heatmaps,
            pred_heatmaps=pred_heatmaps,
            peak_dists=peak_dists,
            all_poses=data["poses"],
            output_path=str(out_path),
        )

        # 汇总
        finite = [d for d in peak_dists if d != float('inf')]
        if finite:
            logger.info(f"  Clip summary: mean_dist={np.mean(finite):.1f}px, "
                         f"median={np.median(finite):.1f}px, <5px: {sum(1 for d in finite if d<5)}/{len(finite)}")

        torch.cuda.empty_cache()

    logger.info(f"\nAll done! Output: {output_dir}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
轨迹热力图可视化
================

沿着一条完整轨迹，可视化每个采样位置的：
- RGB 帧 + GT 热力图叠加
- RGB 帧 + 预测热力图叠加

支持 chunks 数据格式（npz 文件）。

用法:
    python scripts/visualize_trajectory_heatmaps.py \
        --checkpoint /path/to/best.pth \
        --num-clips 3 \
        --frames-per-clip 12 \
        --output-dir ./vis_trajectory \
        --inference-steps 50
"""

import os
import sys
import json
import random
import argparse
import logging
import math
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
from src.data.vln_sliding_window_dataset import compute_history_heatmap

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("vis_traj")


# ==================== build_model (与 train.py 一致) ====================
def build_model(cfg: Dict) -> VLNPipeline:
    model_cfg = cfg['model']
    llm_cfg = model_cfg.get('llm', {})
    heatmap_cfg = model_cfg.get('heatmap_head', {})
    action_cfg = model_cfg.get('action_head', {})
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
        heatmap_cross_attention_levels=tuple(heatmap_cfg['cross_attention_levels']) if heatmap_cfg.get('cross_attention_levels') else None,
        heatmap_num_train_timesteps=heatmap_cfg.get('num_train_timesteps', 100),
        heatmap_cfg_drop_prob=heatmap_cfg.get('cfg_drop_prob', 0.1),
        heatmap_cfg_scale=heatmap_cfg.get('cfg_scale', 3.0),
        heatmap_use_sequence_conditioning=heatmap_cfg.get('use_sequence_conditioning', False),
        heatmap_seq_cross_attn_heads=heatmap_cfg.get('seq_cross_attn_heads', 8),
        heatmap_seq_cross_attn_head_dim=heatmap_cfg.get('seq_cross_attn_head_dim', 64),
        heatmap_use_spatial_injection=heatmap_cfg.get('use_spatial_injection', False),
        heatmap_image_encoder_use_pretrained=heatmap_cfg.get('image_encoder_use_pretrained', False),
        heatmap_sharpen_temperature=heatmap_cfg.get('sharpen_temperature', 0.1),
        heatmap_negative_sample_weight=heatmap_cfg.get('negative_sample_weight', 0.3),
        heatmap_positive_sample_boost=heatmap_cfg.get('positive_sample_boost', 3.0),
        heatmap_peak_spatial_weight=heatmap_cfg.get('peak_spatial_weight', 10.0),
        heatmap_head_type=heatmap_cfg.get('head_type', 'diffusion'),
        dpt_out_channels=heatmap_cfg.get('dpt', {}).get('out_channels', None),
        dpt_features=heatmap_cfg.get('dpt', {}).get('features', 256),
        multi_layer_features=llm_cfg.get('multi_layer_features', False),
        feature_layer_indices=llm_cfg.get('feature_layer_indices', None),
        feature_fusion_method=llm_cfg.get('feature_fusion_method', 'weighted_sum'),
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


# ==================== Chunks 数据加载 ====================

def decode_chunk_rgb(raw: np.ndarray) -> np.ndarray:
    """解码 chunk 中的 JPEG 压缩 RGB 数据"""
    if isinstance(raw, np.ndarray):
        if raw.ndim == 3 and raw.shape[2] >= 3:
            return cv2.cvtColor(raw[:, :, :3], cv2.COLOR_BGR2RGB)
        if raw.ndim == 1:
            if raw.dtype == np.uint8:
                img = cv2.imdecode(raw, cv2.IMREAD_COLOR)
            else:
                arr = np.array(raw, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
            if img is not None:
                return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    if isinstance(raw, (bytes, bytearray)):
        arr = np.frombuffer(raw, dtype=np.uint8)
        img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if img is not None:
            return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return None


def load_clip_chunks(clip_dir: Path) -> Dict:
    """加载一个 clip 的所有 chunk 数据"""
    chunks_dir = clip_dir / "chunks"
    chunk_files = sorted(chunks_dir.glob("chunk_*.npz"))

    all_rgbs = []
    all_poses = []
    all_depths = []

    for cf in chunk_files:
        data = np.load(str(cf), allow_pickle=True)
        keys = set(data.files)

        rgb_key = 'rgb_front' if 'rgb_front' in keys else 'rgb'
        pose_key = 'pose_front' if 'pose_front' in keys else 'pose'
        depth_key = 'depth_front' if 'depth_front' in keys else 'depth'

        n_frames = len(data['frame_ids'])
        for i in range(n_frames):
            all_rgbs.append(data[rgb_key][i])
            all_poses.append(data[pose_key][i].astype(np.float32))
            if depth_key in keys:
                all_depths.append(data[depth_key][i].astype(np.float32))

    # Meta
    meta_file = clip_dir / "meta.json"
    instruction = "Navigate to the destination."
    if meta_file.exists():
        with open(meta_file) as f:
            meta = json.load(f)
        instruction = meta.get("instruction", instruction)

    # Intrinsics
    intrinsics = None
    intrinsics_file = clip_dir / "intrinsics.json"
    if intrinsics_file.exists():
        with open(intrinsics_file) as f:
            intr = json.load(f)
        if "K" in intr:
            intrinsics = np.array(intr["K"], dtype=np.float32)

    return {
        "clip_dir": clip_dir,
        "raw_rgbs": all_rgbs,
        "poses": all_poses,
        "depths": all_depths if all_depths else None,
        "intrinsics": intrinsics,
        "instruction": instruction,
        "n_frames": len(all_rgbs),
    }


def find_good_clips(data_root: str, min_frames: int = 60, num_clips: int = 3, seed: int = 42):
    """找到足够长的 clips"""
    data_root = Path(data_root)
    good_clips = []
    for scene_dir in sorted(data_root.iterdir()):
        if not scene_dir.is_dir():
            continue
        for clip_dir in sorted(scene_dir.iterdir()):
            if not clip_dir.is_dir():
                continue
            chunks_dir = clip_dir / "chunks"
            meta_file = clip_dir / "meta.json"
            if not chunks_dir.exists() or not meta_file.exists():
                continue
            with open(meta_file) as f:
                meta = json.load(f)
            n_frames = int(meta.get("num_frames", 0))
            if n_frames >= min_frames:
                good_clips.append((clip_dir, n_frames))

    random.seed(seed)
    random.shuffle(good_clips)
    return good_clips[:num_clips]


def overlay_heatmap_on_frame(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.5) -> np.ndarray:
    """将热力图叠加到 RGB 帧上 (frame: HWC 0-1, heatmap: HxW 0-1)"""
    H, W = frame.shape[:2]
    hm_resized = cv2.resize(heatmap, (W, H), interpolation=cv2.INTER_CUBIC)

    hm_uint8 = (np.clip(hm_resized, 0, 1) * 255).astype(np.uint8)
    hm_colored = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_INFERNO)
    hm_colored = cv2.cvtColor(hm_colored, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0

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
    """渲染俯视图"""
    positions = []
    for p in all_poses:
        pos = np.array(p, dtype=np.float32)
        x, z = pos[0, 3], pos[2, 3]
        positions.append((x, z))

    positions = np.array(positions)

    x_min, x_max = positions[:, 0].min(), positions[:, 0].max()
    z_min, z_max = positions[:, 1].min(), positions[:, 1].max()

    x_range = max(x_max - x_min, 0.1)
    z_range = max(z_max - z_min, 0.1)
    scale = min((view_w - 2 * margin) / x_range, (view_h - 2 * margin) / z_range)

    def world_to_px(wx, wz):
        px = int(margin + (wx - x_min) * scale)
        py = int(margin + (wz - z_min) * scale)
        py = view_h - py
        return px, py

    canvas = np.ones((view_h, view_w, 3), dtype=np.uint8) * 245

    pts = [world_to_px(x, z) for x, z in positions]
    for i in range(len(pts) - 1):
        cv2.line(canvas, pts[i], pts[i + 1], (180, 180, 180), 1, cv2.LINE_AA)

    cv2.drawMarker(canvas, pts[0], (0, 180, 0), cv2.MARKER_STAR, 16, 2)
    cv2.drawMarker(canvas, pts[-1], (0, 0, 200), cv2.MARKER_SQUARE, 10, 2)

    for idx, (pos_frame, dist) in enumerate(zip(sample_positions, peak_dists)):
        if pos_frame >= len(positions):
            continue
        px, py = world_to_px(positions[pos_frame, 0], positions[pos_frame, 1])

        if dist < 5:
            color = (0, 200, 0)
        elif dist < 15:
            color = (0, 200, 255)
        elif dist == float('inf'):
            color = (200, 200, 200)
        else:
            color = (0, 0, 220)

        cv2.circle(canvas, (px, py), 7, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, (px, py), 7, (0, 0, 0), 1, cv2.LINE_AA)

        pose = np.array(all_poses[pos_frame], dtype=np.float32)
        fwd_world = -pose[:3, 2]
        fwd_xz = np.array([fwd_world[0], fwd_world[2]])
        fwd_len = np.linalg.norm(fwd_xz)
        if fwd_len > 1e-6:
            fwd_xz = fwd_xz / fwd_len
            arrow_len = 18
            dx = int(fwd_xz[0] * arrow_len)
            dy = int(-fwd_xz[1] * arrow_len)
            cv2.arrowedLine(canvas, (px, py), (px + dx, py + dy),
                           color, 2, cv2.LINE_AA, tipLength=0.35)

        label = f"F{pos_frame}"
        cv2.putText(canvas, label, (px + 10, py - 3),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.3, (60, 60, 60), 1, cv2.LINE_AA)

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
    """生成轨迹 grid 可视化"""
    N = len(sample_positions)
    H, W = frames_np[0].shape[:2]

    n_rows = 3
    row_labels = ["RGB Frame", "GT Heatmap", "Pred Heatmap"]

    left_margin = 120
    top_margin = 60
    grid_h = n_rows * H
    grid_w = N * W

    bev_w = max(300, int(grid_h * 0.8))
    bev_h = grid_h

    canvas_w = left_margin + grid_w + 20 + bev_w + 10
    canvas_h = top_margin + grid_h + 10
    canvas = np.ones((canvas_h, canvas_w, 3), dtype=np.uint8) * 255

    title = f"{clip_name} | {instruction[:90]}{'...' if len(instruction) > 90 else ''}"
    cv2.putText(canvas, title, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 1, cv2.LINE_AA)

    for row_idx in range(n_rows):
        y_start = top_margin + row_idx * H
        label_y = y_start + H // 2
        cv2.putText(canvas, row_labels[row_idx], (5, label_y),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

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

            cv2.putText(img_bgr, f"F{pos}", (3, 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

            if row_idx == 2:
                dist_str = f"{dist:.1f}px" if dist != float('inf') else "N/A"
                color = (0, 255, 0) if dist < 5 else (0, 255, 255) if dist < 15 else (0, 0, 255)
                cv2.putText(img_bgr, dist_str, (3, H - 5),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1, cv2.LINE_AA)

            canvas[y_start:y_start + H, x_start:x_start + W] = img_bgr

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
        logger.info(f"  Loaded: {len(state_dict)} params, missing={len(missing)}, unexpected={len(unexpected)}")
    model = model.to(device)
    model.eval()

    # ==================== 找 clips ====================
    min_frames_needed = num_history * stride + 10
    logger.info(f"Finding clips with >= {min_frames_needed} frames...")
    clips = find_good_clips(data_root, min_frames=min_frames_needed,
                            num_clips=args.num_clips, seed=args.seed)
    logger.info(f"  Found {len(clips)} clips")

    if len(clips) == 0:
        logger.error("No clips found! Check data_root and min_frames requirement.")
        return

    # ==================== 处理每个 clip ====================
    for clip_idx, (clip_dir, n_total) in enumerate(clips):
        clip_name = f"{clip_dir.parent.name}/{clip_dir.name}"
        logger.info(f"\n[Clip {clip_idx+1}/{len(clips)}] {clip_name} ({n_total} frames)")

        clip_data = load_clip_chunks(clip_dir)
        n = clip_data["n_frames"]
        logger.info(f"  Loaded {n} frames, instruction: {clip_data['instruction'][:80]}...")

        min_pos = num_history * stride
        max_pos = n - 1
        if max_pos <= min_pos:
            logger.warning(f"  Clip too short ({n} frames, need >{min_pos}), skipping")
            continue

        sample_positions = np.linspace(min_pos, max_pos, args.frames_per_clip, dtype=int).tolist()
        sample_positions = sorted(set(sample_positions))

        logger.info(f"  Sampling {len(sample_positions)} positions: {sample_positions[:5]}...{sample_positions[-3:]}")

        frames_for_vis = []
        gt_heatmaps = []
        pred_heatmaps = []
        peak_dists = []

        for pos_idx, pos in enumerate(sample_positions):
            # 均匀采样历史帧索引 [0, pos)
            history_indices = np.linspace(0, pos - 1, num_history, dtype=int).tolist()
            current_idx = pos

            # 解码当前帧 RGB
            current_rgb = decode_chunk_rgb(clip_data["raw_rgbs"][current_idx])
            if current_rgb is None:
                logger.warning(f"    Failed to decode frame {current_idx}")
                continue
            current_rgb_resized = cv2.resize(current_rgb, image_size)
            current_frame_np = current_rgb_resized.astype(np.float32) / 255.0
            current_frame_tensor = torch.from_numpy(current_frame_np).float().permute(2, 0, 1)

            # 解码历史帧 RGB
            history_tensors = []
            for hi in history_indices:
                rgb = decode_chunk_rgb(clip_data["raw_rgbs"][hi])
                if rgb is None:
                    rgb = current_rgb
                rgb_resized = cv2.resize(rgb, image_size)
                t = torch.from_numpy(rgb_resized).float() / 255.0
                history_tensors.append(t.permute(2, 0, 1))

            # video_frames = history + current (模型期望最后一帧是 current)
            all_frame_tensors = history_tensors + [current_frame_tensor]
            video_frames = torch.stack(all_frame_tensors).unsqueeze(0).to(device)  # [1, T, C, H, W]

            # GT 热力图
            history_poses = [clip_data["poses"][hi] for hi in history_indices]
            current_pose = clip_data["poses"][current_idx]

            current_depth = None
            if clip_data["depths"] is not None and current_idx < len(clip_data["depths"]):
                current_depth = clip_data["depths"][current_idx]

            gt_hm, vis_count = compute_history_heatmap(
                history_poses=history_poses,
                current_pose=current_pose,
                current_depth=current_depth,
                hm_size=hm_size,
                img_size=(640, 480),
                K=clip_data["intrinsics"],
            )

            # 模型推理
            with torch.no_grad(), torch.autocast(device_type='cuda', dtype=torch.bfloat16):
                vis_output = model(
                    video_frames=video_frames,
                    instruction_text=[clip_data["instruction"]],
                    current_observation=video_frames[:, -1],
                    return_heatmaps=True,
                    return_actions=False,
                )

            if 'history_heatmaps' in vis_output and vis_output['history_heatmaps'] is not None:
                pred_hm = vis_output['history_heatmaps'][0, -1].float().cpu().numpy()
                pred_hm = np.clip(pred_hm, 0, 1)
            else:
                pred_hm = np.zeros(hm_size, dtype=np.float32)

            # Peak distance
            if gt_hm.max() > 0.01 and pred_hm.max() > 0.01:
                gt_peak = np.unravel_index(gt_hm.argmax(), gt_hm.shape)
                pred_peak = np.unravel_index(pred_hm.argmax(), pred_hm.shape)
                dist = np.sqrt((gt_peak[0] - pred_peak[0])**2 + (gt_peak[1] - pred_peak[1])**2)
            elif gt_hm.max() <= 0.01 and pred_hm.max() <= 0.01:
                dist = 0.0
            else:
                dist = float('inf')

            frames_for_vis.append(current_frame_np)
            gt_heatmaps.append(gt_hm)
            pred_heatmaps.append(pred_hm)
            peak_dists.append(dist)

            gt_type = "pos" if gt_hm.max() > 0.01 else "neg"
            logger.info(f"    Pos {pos:>4d} [{gt_type}]: peak_dist={dist:>6.1f}px, "
                         f"pred_max={pred_hm.max():.3f}, gt_max={gt_hm.max():.3f}, vis={vis_count}")

        if not frames_for_vis:
            logger.warning(f"  No valid frames for clip, skipping")
            continue

        out_path = output_dir / f"clip_{clip_idx:02d}_{clip_dir.name}.png"
        make_trajectory_grid(
            clip_name=clip_name,
            instruction=clip_data["instruction"],
            sample_positions=sample_positions,
            frames_np=frames_for_vis,
            gt_heatmaps=gt_heatmaps,
            pred_heatmaps=pred_heatmaps,
            peak_dists=peak_dists,
            all_poses=clip_data["poses"],
            output_path=str(out_path),
        )

        finite = [d for d in peak_dists if d != float('inf')]
        if finite:
            logger.info(f"  Clip summary: mean_dist={np.mean(finite):.1f}px, "
                         f"median={np.median(finite):.1f}px, <5px: {sum(1 for d in finite if d<5)}/{len(finite)}")

        torch.cuda.empty_cache()

    logger.info(f"\nAll done! Output: {output_dir}")


if __name__ == "__main__":
    main()

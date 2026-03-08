#!/usr/bin/env python3
"""
轨迹热力图可视化（HeatmapVLN v2）
=================================

沿着一条完整轨迹抽样多个位置，对每个位置聚合所有历史时间步热力图，
并在当前 4 视角中选择 GT 信号最强的视角进行对比可视化。
"""

import argparse
import logging
import random
import sys
from contextlib import nullcontext
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import cv2
import matplotlib
matplotlib.use('Agg')
import numpy as np
import torch
import torch.nn.functional as F
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.data.vln_sliding_window_dataset import VLNSlidingWindowDataset
from src.models.pipeline import VLNPipeline, VLNPipelineConfig

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("vis_traj")

VIEW_NAMES = ["Front", "Right", "Back", "Left"]


def build_model(cfg: Dict, device: str) -> VLNPipeline:
    model_cfg = cfg['model']
    llm_cfg = model_cfg.get('llm', {})
    heatmap_cfg = model_cfg.get('heatmap', {})
    action_cfg = model_cfg.get('action_head', {})
    action_head_type = action_cfg.get('type', 'transformer')

    config = VLNPipelineConfig(
        llm_model_path=llm_cfg.get('model_path', './models/qwen_3.5'),
        llm_hidden_dim=llm_cfg.get('hidden_dim', 4096),
        llm_token_dim=llm_cfg.get('token_dim', 1024),
        llm_torch_dtype=llm_cfg.get('torch_dtype', 'bfloat16'),
        llm_attn_implementation=llm_cfg.get('attn_implementation', 'sdpa'),
        max_video_frames=llm_cfg.get('max_video_frames', 16),
        llm_enable_internal_profiling=llm_cfg.get('enable_internal_profiling', False),
        llm_enable_compile=llm_cfg.get('enable_compile', False),
        llm_compile_mode=llm_cfg.get('compile_mode', 'reduce-overhead'),
        llm_compile_backend=llm_cfg.get('compile_backend', 'inductor'),
        enable_packing=False,
        max_seq_length=llm_cfg.get('max_seq_length', 4096),
        spatial_merge_size=llm_cfg.get('spatial_merge_size', 2),
        device=device,
        enable_heatmap=heatmap_cfg.get('enable', True),
        heatmap_c_vit=heatmap_cfg.get('c_vit', 1152),
        heatmap_c_llm=heatmap_cfg.get('c_llm', 4096),
        heatmap_c_fused=heatmap_cfg.get('c_fused', 256),
        heatmap_vit_layer_indices=heatmap_cfg.get('vit_layer_indices', [6, 12, 18, 24]),
        heatmap_llm_layer_indices=heatmap_cfg.get('llm_layer_indices', [7, 15, 23]),
        heatmap_size=tuple(heatmap_cfg.get('heatmap_size', cfg['data']['init_hm_size'])),
        image_size=heatmap_cfg.get('image_size', cfg['data']['image_size'][0]),
        heatmap_lambda_vis=heatmap_cfg.get('lambda_vis', 1.0),
        heatmap_lambda_pos=heatmap_cfg.get('lambda_pos', 1.0),
        heatmap_lambda_neg=heatmap_cfg.get('lambda_neg', 0.1),
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


def overlay_heatmap_on_frame(frame: np.ndarray, heatmap: np.ndarray, alpha: float = 0.6) -> np.ndarray:
    height, width = frame.shape[:2]
    hm_resized = cv2.resize(heatmap, (width, height), interpolation=cv2.INTER_CUBIC)
    hm_uint8 = (np.clip(hm_resized, 0, 1) * 255).astype(np.uint8)
    hm_color = cv2.applyColorMap(hm_uint8, cv2.COLORMAP_INFERNO)
    hm_color = cv2.cvtColor(hm_color, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    mask = np.clip(hm_resized * 3.0, 0.0, 1.0)[..., None]
    mixed = frame * (1.0 - alpha * mask) + hm_color * (alpha * mask)
    return np.clip(mixed, 0, 1)


def to_bgr_image(image: np.ndarray) -> np.ndarray:
    return cv2.cvtColor((np.clip(image, 0, 1) * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)


def aggregate_history_heatmaps(heatmaps: torch.Tensor) -> torch.Tensor:
    if heatmaps.dim() == 4:
        return heatmaps.max(dim=0).values
    if heatmaps.dim() == 3:
        return heatmaps
    raise ValueError(f"Unsupported heatmap shape: {tuple(heatmaps.shape)}")


def choose_view(gt_heatmaps: np.ndarray, pred_heatmaps: np.ndarray) -> Tuple[int, float]:
    gt_strength = gt_heatmaps.reshape(gt_heatmaps.shape[0], -1).max(axis=1)
    pred_strength = pred_heatmaps.reshape(pred_heatmaps.shape[0], -1).max(axis=1)
    if gt_strength.max() <= 1e-6:
        view_idx = int(pred_strength.argmax())
    else:
        view_idx = int(gt_strength.argmax())

    gt_view = gt_heatmaps[view_idx]
    pred_view = pred_heatmaps[view_idx]
    if gt_view.max() > 0.01 and pred_view.max() > 0.01:
        gt_peak = np.unravel_index(gt_view.argmax(), gt_view.shape)
        pred_peak = np.unravel_index(pred_view.argmax(), pred_view.shape)
        peak_dist = float(np.sqrt((gt_peak[0] - pred_peak[0]) ** 2 + (gt_peak[1] - pred_peak[1]) ** 2))
    elif gt_view.max() <= 0.01 and pred_view.max() <= 0.01:
        peak_dist = 0.0
    else:
        peak_dist = float('inf')

    return view_idx, peak_dist


def wrap_text(text: str, max_chars: int) -> List[str]:
    if len(text) <= max_chars:
        return [text]
    chunks = []
    current = []
    current_len = 0
    for token in text.split():
        extra = 1 if current else 0
        if current_len + len(token) + extra > max_chars:
            chunks.append(" ".join(current))
            current = [token]
            current_len = len(token)
        else:
            current.append(token)
            current_len += len(token) + extra
    if current:
        chunks.append(" ".join(current))
    return chunks[:3]


def make_trajectory_grid(
    clip_name: str,
    instruction: str,
    position_labels: Sequence[str],
    frames_np: Sequence[np.ndarray],
    gt_heatmaps: Sequence[np.ndarray],
    pred_heatmaps: Sequence[np.ndarray],
    chosen_views: Sequence[int],
    peak_dists: Sequence[float],
    output_path: str,
) -> None:
    if not frames_np:
        return

    rows = 3
    cols = len(frames_np)
    tile_h, tile_w = frames_np[0].shape[:2]
    top_margin = 90
    left_margin = 16
    bottom_margin = 20
    canvas = np.full(
        (top_margin + rows * tile_h + bottom_margin, left_margin * 2 + cols * tile_w, 3),
        20,
        dtype=np.uint8,
    )

    cv2.putText(canvas, clip_name[:100], (left_margin, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    y = 52
    for line in wrap_text(instruction, max_chars=max(48, cols * 18)):
        cv2.putText(canvas, line, (left_margin, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (210, 210, 210), 1)
        y += 20

    row_titles = ["View", "GT Overlay", "Pred Overlay"]
    for row_idx, row_title in enumerate(row_titles):
        y_pos = top_margin + row_idx * tile_h + 22
        cv2.putText(canvas, row_title, (4, y_pos), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 220, 220), 1)

    for col_idx in range(cols):
        frame = frames_np[col_idx]
        gt_hm = gt_heatmaps[col_idx]
        pred_hm = pred_heatmaps[col_idx]
        view_idx = chosen_views[col_idx]
        peak_dist = peak_dists[col_idx]
        x_start = left_margin + col_idx * tile_w

        view_img = to_bgr_image(frame)
        gt_img = to_bgr_image(overlay_heatmap_on_frame(frame, gt_hm))
        pred_img = to_bgr_image(overlay_heatmap_on_frame(frame, pred_hm))

        labels = [
            f"{position_labels[col_idx]} | {VIEW_NAMES[view_idx]}",
            f"GT max={gt_hm.max():.3f}",
            f"Pred max={pred_hm.max():.3f}",
        ]
        if peak_dist == float('inf'):
            labels[-1] += " | dist=N/A"
        else:
            labels[-1] += f" | dist={peak_dist:.1f}px"

        for row_idx, img in enumerate((view_img, gt_img, pred_img)):
            y_start = top_margin + row_idx * tile_h
            canvas[y_start:y_start + tile_h, x_start:x_start + tile_w] = img
            cv2.putText(
                canvas,
                labels[row_idx],
                (x_start + 4, y_start + 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.45,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )

    cv2.imwrite(output_path, canvas)
    logger.info(f"  Saved: {output_path}")


def collect_clip_samples(
    dataset: VLNSlidingWindowDataset,
    num_clips: int,
    frames_per_clip: int,
    seed: int,
) -> List[Tuple[int, List[Tuple[int, int]]]]:
    by_clip: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for dataset_idx, (clip_idx, frame_idx) in enumerate(dataset.sample_index):
        by_clip[int(clip_idx)].append((int(frame_idx), int(dataset_idx)))

    clip_items = [(clip_idx, sorted(items, key=lambda x: x[0])) for clip_idx, items in by_clip.items() if items]
    rng = random.Random(seed)
    rng.shuffle(clip_items)

    selected = []
    for clip_idx, items in clip_items[:num_clips]:
        if len(items) <= frames_per_clip:
            sampled = items
        else:
            positions = np.linspace(0, len(items) - 1, frames_per_clip, dtype=int)
            sampled = [items[pos] for pos in sorted(set(positions.tolist()))]
        selected.append((clip_idx, sampled))
    return selected


def infer_sample(model: VLNPipeline, sample: Dict[str, Any], device: torch.device) -> np.ndarray:
    history_frames = sample['history_frames'].unsqueeze(0).to(device)
    current_frame = sample['current_frame'].unsqueeze(0).to(device)
    current_views = sample.get('current_views')
    history_panoramas = sample.get('history_panoramas')
    if current_views is None or history_panoramas is None:
        raise ValueError("当前数据集不包含全景 current_views/history_panoramas，无法运行 v2 热力图可视化。")

    current_views = current_views.unsqueeze(0).to(device)
    history_panoramas = history_panoramas.unsqueeze(0).to(device)
    video_frames = torch.cat([history_frames, history_frames[:, -1:]], dim=1)
    autocast_context = (
        torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        if device.type == 'cuda'
        else nullcontext()
    )

    with torch.no_grad(), autocast_context:
        outputs = model(
            video_frames=video_frames,
            instruction_text=[sample['text']],
            current_observation=current_frame,
            current_views=current_views,
            history_panoramas=history_panoramas,
            return_heatmaps=True,
            return_actions=False,
        )

    heatmaps = outputs.get('heatmaps')
    if heatmaps is None:
        raise RuntimeError("模型未返回 heatmaps。")
    return heatmaps[0].float().cpu()


def main() -> int:
    parser = argparse.ArgumentParser(description="轨迹热力图可视化 (HeatmapVLN v2)")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num-clips', type=int, default=3, help='可视化 clip 数')
    parser.add_argument('--frames-per-clip', type=int, default=12, help='每个 clip 抽样的位置数')
    parser.add_argument('--output-dir', type=str, default='./vis_trajectory')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', type=str, default=None, help='默认使用配置里的 val_split')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    cfg = ckpt['config']
    logger.info(f"  Epoch: {ckpt.get('epoch', '?')}")

    split = args.split or cfg['data'].get('val_split', 'val')
    sw_cfg = cfg['data']['sliding_window']

    logger.info("Loading dataset...")
    dataset = VLNSlidingWindowDataset(
        root=cfg['data']['root'],
        split=split,
        min_history=sw_cfg['min_history'],
        num_history_sample=sw_cfg['num_history_sample'],
        image_size=tuple(cfg['data']['image_size']),
        hm_size=tuple(cfg['data']['init_hm_size']),
        load_depth=sw_cfg.get('load_depth', True),
        cache_poses=sw_cfg.get('cache_poses', True),
        sample_stride=sw_cfg.get('sample_stride', 2),
        clip_level_sampling=False,
        samples_per_clip=sw_cfg.get('val_samples_per_clip', 2),
        enable_augmentation=False,
        defer_heatmap_to_gpu=False,
        load_history_frames=sw_cfg.get('load_history_frames', True),
    )

    if not getattr(dataset, '_is_panoramic', False):
        logger.error("当前数据集不是全景 4 视角格式，无法运行 v2 轨迹热力图可视化。")
        return 1

    selected_clips = collect_clip_samples(
        dataset=dataset,
        num_clips=args.num_clips,
        frames_per_clip=args.frames_per_clip,
        seed=args.seed,
    )
    if not selected_clips:
        logger.error("没有找到可用样本。")
        return 1

    logger.info("Building model...")
    model = build_model(cfg, device=str(device))
    state_dict = ckpt.get('trainable_state_dict', ckpt.get('model_state_dict', {}))
    if state_dict:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"  Loaded: {len(state_dict)} params, missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        logger.warning("  No trainable weights in checkpoint!")
    model = model.to(device)
    model.eval()

    logger.info(f"Visualizing {len(selected_clips)} clips...")
    for clip_order, (clip_idx, items) in enumerate(selected_clips):
        clip_dir = dataset.clips[clip_idx]
        clip_name = f"{clip_dir.parent.name}/{clip_dir.name}"
        logger.info(f"[Clip {clip_order + 1}/{len(selected_clips)}] {clip_name} ({len(items)} samples)")

        frames_np: List[np.ndarray] = []
        gt_heatmaps: List[np.ndarray] = []
        pred_heatmaps: List[np.ndarray] = []
        chosen_views: List[int] = []
        peak_dists: List[float] = []
        position_labels: List[str] = []
        instruction = ""

        for frame_idx, dataset_idx in items:
            sample = dataset[dataset_idx]
            instruction = sample['text']

            pred_all = infer_sample(model, sample, device=device)
            gt_all = sample['heatmap'].float().cpu()

            pred_agg = aggregate_history_heatmaps(pred_all)
            gt_agg = aggregate_history_heatmaps(gt_all)

            if pred_agg.shape[-2:] != gt_agg.shape[-2:]:
                pred_agg = F.interpolate(
                    pred_agg.unsqueeze(0),
                    size=gt_agg.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                ).squeeze(0)

            pred_np = np.clip(pred_agg.numpy(), 0, 1)
            gt_np = np.clip(gt_agg.numpy(), 0, 1)
            current_views_np = sample['current_views'].cpu().numpy().transpose(0, 2, 3, 1)

            view_idx, peak_dist = choose_view(gt_np, pred_np)
            frames_np.append(np.clip(current_views_np[view_idx], 0, 1))
            gt_heatmaps.append(gt_np[view_idx])
            pred_heatmaps.append(pred_np[view_idx])
            chosen_views.append(view_idx)
            peak_dists.append(peak_dist)
            position_labels.append(f"F{frame_idx}")

            logger.info(
                "  frame=%4d view=%s peak_dist=%s gt_max=%.3f pred_max=%.3f",
                frame_idx,
                VIEW_NAMES[view_idx],
                "N/A" if peak_dist == float('inf') else f"{peak_dist:.1f}px",
                float(gt_np[view_idx].max()),
                float(pred_np[view_idx].max()),
            )

        out_path = output_dir / f"clip_{clip_order:02d}_{clip_dir.name}.png"
        make_trajectory_grid(
            clip_name=clip_name,
            instruction=instruction,
            position_labels=position_labels,
            frames_np=frames_np,
            gt_heatmaps=gt_heatmaps,
            pred_heatmaps=pred_heatmaps,
            chosen_views=chosen_views,
            peak_dists=peak_dists,
            output_path=str(out_path),
        )

        finite = [dist for dist in peak_dists if dist != float('inf')]
        if finite:
            logger.info(
                "  Summary: mean_dist=%.1fpx, median=%.1fpx, <5px=%d/%d",
                float(np.mean(finite)),
                float(np.median(finite)),
                sum(1 for dist in finite if dist < 5),
                len(finite),
            )
        torch.cuda.empty_cache()

    logger.info(f"All done! Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

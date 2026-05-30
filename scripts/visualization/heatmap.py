#!/usr/bin/env python3
"""
Heatmap 推理可视化 (v2 — 全景 4 视角)
========================================

加载训练好的模型权重，使用与训练完全相同的数据加载流程，
进行推理，生成 4 视角 predicted heatmap vs GT heatmap 的对比可视化。

每个样本输出一张大图：
    4 行（front / right / back / left）× 4 列（视角图 | GT 热力图 | Pred 热力图 | Overlay）

用法:
    python scripts/run.py visualize heatmap \
        --checkpoint /root/autodl-tmp/heatmap_training_outputs/run_.../ckpts/best.pth \
        --num-samples 10 \
        --output-dir ./vis_heatmap_4view
"""

import argparse
import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np
import torch
import torch.nn.functional as F

matplotlib.use('Agg')
import cv2
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.training.collate import collate_fn
from scripts.training.model_builder import build_model
from scripts.training.utils import make_autocast_context, load_checkpoint

from src.data.factory import build_sliding_window_dataset
from src.utils.gpu_heatmap import GPUHeatmapComputer

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("visualize")

VIEW_NAMES = ["Front", "Right", "Back", "Left"]


# ==================== 4-View Panoramic Visualization ====================
def visualize_panoramic_sample(
    sample_idx: int,
    current_views: np.ndarray,
    pred_heatmaps: np.ndarray,
    gt_heatmaps: np.ndarray,
    visibility: np.ndarray | None,
    instruction: str,
    output_path: str,
    metrics: dict | None = None,
):
    """
    全景 4 视角可视化。

    Args:
        current_views:  (4, H, W, 3) — 4 个方向的 RGB 图
        pred_heatmaps:  (4, Hm, Wm) — 4 个方向的预测热力图
        gt_heatmaps:    (4, Hm, Wm) — 4 个方向的 GT 热力图
        visibility:     (4,) — 4 个方向的可见性 logit（可选）
        instruction:    导航指令文本
        output_path:    输出文件路径
        metrics:        指标字典
    """
    n_views = 4
    # 4 rows x 4 cols: View | GT | Pred | Overlay
    fig, axes = plt.subplots(n_views, 4, figsize=(20, 5 * n_views))

    for v in range(n_views):
        rgb = np.clip(current_views[v], 0, 1)
        gt_hm = gt_heatmaps[v]
        pred_hm = np.clip(pred_heatmaps[v], 0, 1)

        vis_str = ""
        if visibility is not None:
            vis_val = visibility[v]
            vis_str = f" (vis={vis_val:.2f})"

        # Col 0: View
        axes[v, 0].imshow(rgb)
        axes[v, 0].set_title(f"{VIEW_NAMES[v]}{vis_str}", fontsize=12, fontweight='bold')
        axes[v, 0].axis('off')

        # Col 1: GT Heatmap
        axes[v, 1].imshow(gt_hm, cmap='inferno', vmin=0, vmax=max(gt_hm.max(), 0.01))
        axes[v, 1].set_title(f"GT (max={gt_hm.max():.3f})", fontsize=11)
        axes[v, 1].axis('off')
        if gt_hm.max() > 0.01:
            gy, gx = np.unravel_index(gt_hm.argmax(), gt_hm.shape)
            axes[v, 1].plot(gx, gy, 'g+', markersize=12, markeredgewidth=2)

        # Col 2: Pred Heatmap
        axes[v, 2].imshow(pred_hm, cmap='inferno', vmin=0, vmax=max(pred_hm.max(), 0.01))
        axes[v, 2].set_title(f"Pred (max={pred_hm.max():.3f})", fontsize=11)
        axes[v, 2].axis('off')
        if pred_hm.max() > 0.01:
            py, px = np.unravel_index(pred_hm.argmax(), pred_hm.shape)
            axes[v, 2].plot(px, py, 'r+', markersize=12, markeredgewidth=2)

        # Col 3: Overlay
        H, W = rgb.shape[:2]
        pred_up = cv2.resize(pred_hm, (W, H), interpolation=cv2.INTER_CUBIC)
        pred_color = plt.cm.inferno(pred_up / max(pred_up.max(), 0.001))[:, :, :3]
        alpha = np.clip(pred_up * 3, 0, 0.7)[:, :, None]
        overlay = rgb * (1 - alpha) + pred_color * alpha
        overlay = np.clip(overlay, 0, 1)
        axes[v, 3].imshow(overlay)
        axes[v, 3].set_title("Overlay", fontsize=11)
        axes[v, 3].axis('off')

    # Title with instruction
    title_text = f"Sample {sample_idx}"
    if metrics:
        summary_parts = []
        for k, val in metrics.items():
            summary_parts.append(f"{k}={val}")
        title_text += "  |  " + ", ".join(summary_parts[:4])
    instr_short = instruction[:120] + ("..." if len(instruction) > 120 else "")
    fig.suptitle(f"{title_text}\n{instr_short}", fontsize=13, fontweight='bold')
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(output_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description="Heatmap 推理可视化 (4-view panoramic)")
    parser.add_argument('--checkpoint', type=str, required=True, help='模型 checkpoint 路径')
    parser.add_argument('--num-samples', type=int, default=10, help='可视化样本数')
    parser.add_argument('--output-dir', type=str, default='./vis_heatmap_4view', help='输出目录')
    parser.add_argument('--device', type=str, default='cuda:0')
    args = parser.parse_args()

    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # ==================== 加载 checkpoint ====================
    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint, map_location='cpu', trust_checkpoint=True)
    cfg = ckpt['config']
    epoch = ckpt.get('epoch', '?')
    best_val = ckpt.get('best_val_loss', '?')
    logger.info(f"  Epoch: {epoch}, Best val loss: {best_val}")

    # ==================== 构建数据集 ====================
    logger.info("Loading dataset...")
    sw_cfg = cfg['data']['sliding_window']
    defer_heatmap_to_gpu = sw_cfg.get('defer_heatmap_to_gpu', False)

    val_dataset = build_sliding_window_dataset(
        cfg,
        split=cfg['data'].get('val_split', 'val'),
        samples_per_clip=sw_cfg.get('val_samples_per_clip', 2),
        defer_heatmap_to_gpu=defer_heatmap_to_gpu,
    )
    if hasattr(val_dataset, 'set_epoch'):
        val_dataset.set_epoch(0)

    logger.info(f"  Val dataset: {len(val_dataset)} samples")

    from torch.utils.data import DataLoader
    val_loader = DataLoader(
        val_dataset, batch_size=1, shuffle=False,
        collate_fn=collate_fn, num_workers=0,
    )

    # ==================== GPU 热力图计算器 ====================
    gpu_heatmap_computer = None
    if defer_heatmap_to_gpu:
        hm_size = tuple(cfg['data']['init_hm_size'])
        gpu_heatmap_computer = GPUHeatmapComputer(
            hm_size=hm_size,
            img_size=(640, 480),
            device=device,
        )
        logger.info(f"  GPU heatmap computer enabled (hm_size={hm_size})")

    # ==================== 构建模型 ====================
    logger.info("Building model...")
    model = build_model(cfg, verbose=False, enable_action_head=False)

    state_dict = ckpt.get('trainable_state_dict', ckpt.get('model_state_dict', {}))
    if state_dict:
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        logger.info(f"  Loaded: {len(state_dict)} params, missing={len(missing)}, unexpected={len(unexpected)}")
    else:
        logger.warning("  No trainable weights in checkpoint!")

    model = model.to(device)
    model.eval()
    logger.info("Model ready.")

    # ==================== 推理 + 可视化 ====================
    logger.info(f"Running inference on {args.num_samples} samples...")
    all_peak_dists = []
    all_ious = []
    count = 0

    for i, batch in enumerate(val_loader):
        if count >= args.num_samples:
            break

        history_frames = batch['history_frames']
        current_frame = batch['current_frame']
        text = batch['text']

        # GT 热力图
        if gpu_heatmap_computer is not None and 'history_poses' in batch and 'current_views' not in batch:
            history_poses = batch['history_poses'].to(device)
            current_poses = batch['current_pose'].to(device)
            has_depth = batch.get('has_depth', False)
            current_depths = batch['current_depth'].to(device) if has_depth else None
            has_intrinsics = batch.get('has_intrinsics', False)
            intrinsics = batch['intrinsics'].to(device) if has_intrinsics else None
            gt_heatmap = gpu_heatmap_computer.compute_batch(
                history_poses=history_poses,
                current_poses=current_poses,
                current_depths=current_depths,
                intrinsics=intrinsics,
            )
        else:
            gt_heatmap = batch['heatmap'].to(device)

        # 检查是否有全景数据
        has_panoramic = 'current_views' in batch and 'history_panoramas' in batch
        current_views = batch.get('current_views')
        history_panoramas = batch.get('history_panoramas')

        if current_views is not None:
            current_views = current_views.to(device)
        if history_panoramas is not None:
            history_panoramas = history_panoramas.to(device)

        # 跳过 GT 全空的样本
        if gt_heatmap.max() < 0.01:
            continue

        video_frames = torch.cat([history_frames, history_frames[:, -1:]], dim=1)
        instruction_text = list(text) if text else None

        with torch.no_grad(), make_autocast_context(device, cfg.get('optim', {}).get('amp', 'bf16')):
            vis_output = model(
                video_frames=video_frames,
                instruction_text=instruction_text,
                current_observation=current_frame.to(device),
                current_views=current_views,
                history_panoramas=history_panoramas,
                return_heatmaps=True,
                return_actions=False,
            )

        if 'heatmaps' not in vis_output or vis_output['heatmaps'] is None:
            logger.warning(f"  Sample {i}: no heatmap output")
            continue

        # pred shape: (B, N_hist, 4, 64, 64) — 取第一个历史位置
        pred_all = vis_output['heatmaps']  # (B, N_hist, 4, H, W)
        visibility = vis_output.get('visibility')  # (B, N_hist, 4)

        if pred_all.dim() == 5:
            pred_4view = pred_all[0, 0]  # (4, H, W)
            vis_4 = visibility[0, 0].float().cpu().numpy() if visibility is not None else None
        elif pred_all.dim() == 4 and pred_all.shape[1] == 4:
            pred_4view = pred_all[0]  # (4, H, W)
            vis_4 = visibility[0].float().cpu().numpy() if visibility is not None and visibility.dim() >= 2 else None
        else:
            pred_4view = pred_all[0].unsqueeze(0).expand(4, -1, -1)
            vis_4 = None

        # GT shape: (B, N_hist, 4, Hm, Wm) or (B, 4, Hm, Wm) or (B, Hm, Wm)
        gt = gt_heatmap[0]
        if gt.dim() == 3 and gt.shape[0] == 4:
            gt_4view = gt  # (4, Hm, Wm)
        elif gt.dim() == 4:
            gt_4view = gt[0]  # (4, Hm, Wm)  first history position
        else:
            gt_4view = gt.unsqueeze(0).expand(4, -1, -1)

        # Resize pred to match GT if needed
        if pred_4view.shape[-2:] != gt_4view.shape[-2:]:
            pred_4view = F.interpolate(
                pred_4view.unsqueeze(0), size=gt_4view.shape[-2:],
                mode='bilinear', align_corners=False,
            ).squeeze(0)

        # Convert to numpy
        pred_np = pred_4view.detach().float().cpu().numpy()  # (4, Hm, Wm)
        gt_np = gt_4view.float().cpu().numpy()                # (4, Hm, Wm)

        # Current views: (B, 4, C, H, W) -> (4, H, W, C)
        if has_panoramic:
            views_np = batch['current_views'][0].cpu().numpy().transpose(0, 2, 3, 1)  # (4, H, W, 3)
        else:
            cf_np = current_frame[0].cpu().numpy().transpose(1, 2, 0)  # (H, W, 3)
            views_np = np.stack([cf_np] * 4, axis=0)

        # Per-view metrics
        best_view = -1
        best_peak_dist = float('inf')
        for v in range(4):
            gt_v = gt_np[v]
            pred_v = pred_np[v]
            if gt_v.max() > 0.01 and pred_v.max() > 0.01:
                gt_peak = np.unravel_index(gt_v.argmax(), gt_v.shape)
                pred_peak = np.unravel_index(pred_v.argmax(), pred_v.shape)
                dist = np.sqrt((gt_peak[0] - pred_peak[0])**2 + (gt_peak[1] - pred_peak[1])**2)
                if dist < best_peak_dist:
                    best_peak_dist = dist
                    best_view = v

        # Summary IoU across all views
        gt_mask = gt_np > 0.5 * max(gt_np.max(), 0.01)
        pred_mask = pred_np > 0.5 * max(pred_np.max(), 0.01)
        intersection = (gt_mask & pred_mask).sum()
        union = (gt_mask | pred_mask).sum()
        iou = intersection / max(union, 1)

        mse = np.mean((gt_np - np.clip(pred_np, 0, 1)) ** 2)

        all_peak_dists.append(best_peak_dist)
        all_ious.append(iou)

        metrics = {
            'peak_dist': f"{best_peak_dist:.1f}px",
            'best_view': VIEW_NAMES[best_view] if best_view >= 0 else "N/A",
            'iou': f"{iou:.3f}",
            'mse': f"{mse:.4f}",
        }

        instr = text[0] if text else "N/A"
        logger.info(
            f"  [{count+1:>2}] peak_dist={best_peak_dist:>5.1f}px ({VIEW_NAMES[best_view] if best_view >= 0 else 'N/A'}), "
            f"iou={iou:.3f}, pred_max={pred_np.max():.3f}, gt_max={gt_np.max():.3f}"
        )

        out_path = output_dir / f"sample_{count:03d}.png"
        visualize_panoramic_sample(
            count + 1, views_np, pred_np, gt_np,
            vis_4, instr, str(out_path), metrics=metrics,
        )

        count += 1
        torch.cuda.empty_cache()

    # ==================== 汇总 ====================
    logger.info("")
    logger.info("=" * 60)
    logger.info(f"SUMMARY (Epoch {epoch}, {count} samples)")
    logger.info("=" * 60)

    finite_dists = [d for d in all_peak_dists if d != float('inf')]
    if finite_dists:
        hm_size = cfg['data']['init_hm_size']
        logger.info(f"Peak Distance ({hm_size[0]}x{hm_size[1]}):")
        logger.info(f"  Mean:   {np.mean(finite_dists):.1f} px")
        logger.info(f"  Median: {np.median(finite_dists):.1f} px")
        logger.info(f"  Min:    {np.min(finite_dists):.1f} px")
        logger.info(f"  Max:    {np.max(finite_dists):.1f} px")
        logger.info(f"  < 5px:  {sum(1 for d in finite_dists if d < 5)}/{len(finite_dists)}")
        logger.info(f"  < 10px: {sum(1 for d in finite_dists if d < 10)}/{len(finite_dists)}")

    if all_ious:
        logger.info("IoU:")
        logger.info(f"  Mean:   {np.mean(all_ious):.3f}")
        logger.info(f"  Median: {np.median(all_ious):.3f}")

    inf_count = sum(1 for d in all_peak_dists if d == float('inf'))
    if inf_count:
        logger.info(f"Empty predictions: {inf_count}/{len(all_peak_dists)}")

    logger.info(f"\nOutput: {output_dir}")
    logger.info("=" * 60)


if __name__ == "__main__":
    main()

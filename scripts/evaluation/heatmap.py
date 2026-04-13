#!/usr/bin/env python3
"""
热力图质量评估脚本
==================

使用 best checkpoint 对验证集进行完整扩散推理，
计算热力图质量指标并生成可视化对比图。

支持 defer_heatmap_to_gpu 模式（在 GPU 上计算 GT 热力图）。

用法:
    python scripts/run.py evaluate heatmap \
        --config configs/train_heatmap_config.yaml \
        --checkpoint /path/to/best.pth \
        --max-samples 100 \
        --num-vis 20
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np
import torch
import yaml

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from scripts.train import build_model

from src.data.factory import build_sliding_window_dataset
from src.utils.gpu_heatmap import GPUHeatmapComputer

logging.basicConfig(
    level=logging.INFO,
    format='[%(asctime)s] [%(levelname)s] %(message)s',
    datefmt='%H:%M:%S'
)
logger = logging.getLogger("eval_heatmap")


def load_checkpoint(ckpt_path: str, model: torch.nn.Module, device: torch.device):
    """加载 checkpoint（支持 trainable_state_dict 格式）"""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    state_dict = ckpt.get('trainable_state_dict', ckpt.get('model_state_dict', ckpt))

    if isinstance(state_dict, dict) and len(state_dict) > 0:
        first_key = next(iter(state_dict.keys()))
        if first_key.startswith('module.'):
            state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}

    missing, unexpected = model.load_state_dict(state_dict, strict=False)

    epoch = ckpt.get('epoch', 'N/A')
    stage = ckpt.get('stage_name', 'N/A')
    best_val = ckpt.get('best_val_loss', 'N/A')

    logger.info(f"Checkpoint loaded: epoch={epoch}, stage={stage}, best_val_loss={best_val}")
    logger.info(f"  Loaded {len(state_dict)} params, missing={len(missing)}, unexpected={len(unexpected)}")

    return model


def should_use_gpu_gt(batch: dict[str, Any]) -> bool:
    return 'history_poses' in batch and 'current_views' not in batch


def flatten_heatmap_slices(heatmaps: torch.Tensor) -> torch.Tensor:
    if heatmaps.dim() <= 3:
        return heatmaps
    return heatmaps.reshape(-1, heatmaps.shape[-2], heatmaps.shape[-1])


def select_primary_heatmap_slice(heatmaps: torch.Tensor) -> torch.Tensor:
    if heatmaps.dim() == 5:
        return heatmaps[:, 0, 0]
    if heatmaps.dim() == 4 and heatmaps.shape[1] == 4:
        return heatmaps[:, 0]
    if heatmaps.dim() == 4:
        return heatmaps[:, -1]
    return heatmaps


def compute_metrics(pred_hm: np.ndarray, gt_hm: np.ndarray) -> dict[str, float]:
    """计算单个热力图的质量指标"""
    # 检查 GT 是否为空
    gt_max = gt_hm.max()
    if gt_max < 1e-6:
        return {
            'gt_is_empty': True,
            'pred_max': float(pred_hm.max()),
            'false_positive_energy': float((pred_hm ** 2).mean()),
        }

    # 归一化
    pred_norm = pred_hm / (pred_hm.max() + 1e-8)
    gt_norm = gt_hm / (gt_hm.max() + 1e-8)

    # 1. Peak location error (像素距离)
    pred_peak = np.unravel_index(np.argmax(pred_hm), pred_hm.shape)
    gt_peak = np.unravel_index(np.argmax(gt_hm), gt_hm.shape)
    peak_error = np.sqrt((pred_peak[0] - gt_peak[0])**2 + (pred_peak[1] - gt_peak[1])**2)

    # 2. IoU at multiple thresholds
    ious = {}
    for thresh in [0.1, 0.3, 0.5]:
        pred_mask = pred_norm > thresh
        gt_mask = gt_norm > thresh
        intersection = (pred_mask & gt_mask).sum()
        union = (pred_mask | gt_mask).sum()
        ious[f'iou_{thresh}'] = float(intersection / (union + 1e-6))

    # 3. MSE
    mse = float(((pred_norm - gt_norm) ** 2).mean())

    # 4. Cosine similarity
    cos_sim = float(np.dot(pred_norm.flatten(), gt_norm.flatten()) / (
        np.linalg.norm(pred_norm) * np.linalg.norm(gt_norm) + 1e-8
    ))

    # 5. Pred max and GT max
    pred_max = float(pred_hm.max())

    return {
        'gt_is_empty': False,
        'peak_error': peak_error,
        'mse': mse,
        'cosine_sim': cos_sim,
        'pred_max': pred_max,
        'gt_max': float(gt_max),
        **ious,
    }


def visualize_batch(
    current_frames: torch.Tensor,
    gt_heatmaps: torch.Tensor,
    pred_heatmaps: torch.Tensor,
    save_path: Path,
    batch_idx: int,
    metrics_list: list,
    num_samples: int = 4,
):
    """生成单个 batch 的可视化对比图"""
    B = min(num_samples, current_frames.shape[0])

    fig, axes = plt.subplots(B, 4, figsize=(16, 4 * B))
    if B == 1:
        axes = axes.reshape(1, -1)

    for i in range(B):
        # Input frame
        rgb = current_frames[i].cpu().numpy().transpose(1, 2, 0)
        rgb = np.clip(rgb, 0, 1)
        axes[i, 0].imshow(rgb)
        axes[i, 0].set_title("Input Frame")
        axes[i, 0].axis('off')

        # GT heatmap
        gt_hm = gt_heatmaps[i].cpu().numpy()
        im1 = axes[i, 1].imshow(gt_hm, cmap='inferno', vmin=0, vmax=max(gt_hm.max(), 0.01))
        axes[i, 1].set_title(f"GT (max={gt_hm.max():.3f})")
        axes[i, 1].axis('off')
        plt.colorbar(im1, ax=axes[i, 1], fraction=0.046)

        # Pred heatmap
        pred_hm = pred_heatmaps[i].cpu().numpy()
        pred_hm = np.clip(pred_hm, 0, 1)
        im2 = axes[i, 2].imshow(pred_hm, cmap='inferno', vmin=0, vmax=max(pred_hm.max(), 0.01))
        axes[i, 2].set_title(f"Pred (max={pred_hm.max():.3f})")
        axes[i, 2].axis('off')
        plt.colorbar(im2, ax=axes[i, 2], fraction=0.046)

        # Overlay: GT contour on pred
        axes[i, 3].imshow(rgb)
        if gt_hm.max() > 0.01:
            # Resize heatmaps to image size for overlay
            h, w = rgb.shape[:2]
            gt_resized = np.array(
                torch.nn.functional.interpolate(
                    torch.from_numpy(gt_hm).unsqueeze(0).unsqueeze(0).float(),
                    size=(h, w), mode='bilinear', align_corners=False
                ).squeeze().numpy()
            )
            pred_resized = np.array(
                torch.nn.functional.interpolate(
                    torch.from_numpy(pred_hm).unsqueeze(0).unsqueeze(0).float(),
                    size=(h, w), mode='bilinear', align_corners=False
                ).squeeze().numpy()
            )
            axes[i, 3].contour(gt_resized, levels=[0.3], colors=['lime'], linewidths=2)
            axes[i, 3].contour(pred_resized, levels=[0.3], colors=['red'], linewidths=2)

            # Mark peaks
            gt_peak = np.unravel_index(np.argmax(gt_resized), gt_resized.shape)
            pred_peak = np.unravel_index(np.argmax(pred_resized), pred_resized.shape)
            axes[i, 3].plot(gt_peak[1], gt_peak[0], 'g+', markersize=15, markeredgewidth=3)
            axes[i, 3].plot(pred_peak[1], pred_peak[0], 'r+', markersize=15, markeredgewidth=3)

        m = metrics_list[i] if i < len(metrics_list) else {}
        info = ""
        if not m.get('gt_is_empty', True):
            info = f"peak_err={m.get('peak_error', 0):.1f}px  IoU@0.3={m.get('iou_0.3', 0):.3f}"
        else:
            info = f"GT empty, pred_max={m.get('pred_max', 0):.3f}"
        axes[i, 3].set_title(f"Overlay\n{info}", fontsize=9)
        axes[i, 3].axis('off')

    plt.suptitle(f"Batch {batch_idx}", fontsize=14)
    plt.tight_layout()
    plt.savefig(save_path, dpi=120, bbox_inches='tight')
    plt.close(fig)


@torch.no_grad()
def evaluate_heatmap(
    model,
    dataloader: DataLoader,
    gpu_heatmap_computer: GPUHeatmapComputer,
    device: torch.device,
    save_dir: Path,
    max_samples: int = 200,
    num_vis: int = 20,
):
    """运行完整的热力图评估"""
    model.eval()

    all_metrics = []
    nonempty_metrics = []
    empty_metrics = []

    num_batches = 0
    vis_dir = save_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)

    pbar = tqdm(dataloader, desc="Evaluating heatmap quality")

    for batch in pbar:
        if max_samples and num_batches * dataloader.batch_size >= max_samples:
            break

        num_batches += 1

        # 获取当前帧
        current_frame = batch['current_frame']
        B = current_frame.shape[0]

        # GPU 计算 GT 热力图
        if should_use_gpu_gt(batch):
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

        history_frames = batch['history_frames']
        video_frames = torch.cat([
            history_frames,
            history_frames[:, -1:]
        ], dim=1).to(device)
        text = batch['text']
        instruction = list(text) if text else None
        current_views = batch.get('current_views')
        history_panoramas = batch.get('history_panoramas')
        if current_views is not None:
            current_views = current_views.to(device)
        if history_panoramas is not None:
            history_panoramas = history_panoramas.to(device)

        # 模型推理（完整扩散）
        with torch.autocast(device_type='cuda', dtype=torch.bfloat16):
            output = model(
                video_frames=video_frames,
                instruction_text=instruction,
                current_observation=current_frame.to(device),
                current_views=current_views,
                history_panoramas=history_panoramas,
                return_heatmaps=True,
            )

        # 获取预测热力图
        pred_heatmap = output.get('heatmaps')
        if pred_heatmap is None:
            continue

        pred_heatmap = pred_heatmap.float()

        metric_pred_heatmap = flatten_heatmap_slices(pred_heatmap)
        metric_gt_heatmap = flatten_heatmap_slices(gt_heatmap)

        # 计算指标
        batch_metrics = []
        if metric_pred_heatmap.shape != metric_gt_heatmap.shape:
            logger.warning(
                "Skip batch %d due to heatmap shape mismatch: pred=%s gt=%s",
                num_batches,
                tuple(metric_pred_heatmap.shape),
                tuple(metric_gt_heatmap.shape),
            )
            continue

        for b in range(metric_pred_heatmap.shape[0]):
            m = compute_metrics(
                metric_pred_heatmap[b].cpu().numpy(),
                metric_gt_heatmap[b].cpu().numpy()
            )
            all_metrics.append(m)
            batch_metrics.append(m)

            if m['gt_is_empty']:
                empty_metrics.append(m)
            else:
                nonempty_metrics.append(m)

        # 可视化
        if num_batches <= num_vis:
            vis_path = vis_dir / f"batch_{num_batches:04d}.png"
            visualize_batch(
                current_frame,
                select_primary_heatmap_slice(gt_heatmap),
                select_primary_heatmap_slice(pred_heatmap),
                vis_path, num_batches, batch_metrics,
                num_samples=min(4, B),
            )

        # 更新 progress bar
        if nonempty_metrics:
            avg_peak = np.mean([m['peak_error'] for m in nonempty_metrics])
            avg_iou = np.mean([m['iou_0.3'] for m in nonempty_metrics])
            pbar.set_postfix({
                'peak_err': f"{avg_peak:.1f}",
                'iou@0.3': f"{avg_iou:.3f}",
                'empty': f"{len(empty_metrics)}/{len(all_metrics)}",
            })

    # 汇总结果
    results = {
        'total_samples': len(all_metrics),
        'nonempty_samples': len(nonempty_metrics),
        'empty_samples': len(empty_metrics),
        'empty_ratio': len(empty_metrics) / max(len(all_metrics), 1),
    }

    if nonempty_metrics:
        results['nonempty'] = {
            'peak_error_mean': float(np.mean([m['peak_error'] for m in nonempty_metrics])),
            'peak_error_median': float(np.median([m['peak_error'] for m in nonempty_metrics])),
            'peak_error_std': float(np.std([m['peak_error'] for m in nonempty_metrics])),
            'mse_mean': float(np.mean([m['mse'] for m in nonempty_metrics])),
            'cosine_sim_mean': float(np.mean([m['cosine_sim'] for m in nonempty_metrics])),
            'iou_0.1_mean': float(np.mean([m['iou_0.1'] for m in nonempty_metrics])),
            'iou_0.3_mean': float(np.mean([m['iou_0.3'] for m in nonempty_metrics])),
            'iou_0.5_mean': float(np.mean([m['iou_0.5'] for m in nonempty_metrics])),
            'pred_max_mean': float(np.mean([m['pred_max'] for m in nonempty_metrics])),
            'gt_max_mean': float(np.mean([m['gt_max'] for m in nonempty_metrics])),
        }

    if empty_metrics:
        results['empty'] = {
            'false_positive_energy_mean': float(np.mean([m['false_positive_energy'] for m in empty_metrics])),
            'pred_max_mean': float(np.mean([m['pred_max'] for m in empty_metrics])),
            'pred_max_max': float(np.max([m['pred_max'] for m in empty_metrics])),
        }

    return results


def main():
    parser = argparse.ArgumentParser(description='Heatmap Quality Evaluation')
    parser.add_argument('--config', type=str, default='configs/train_heatmap_config.yaml')
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--max-samples', type=int, default=200)
    parser.add_argument('--num-vis', type=int, default=20)
    parser.add_argument('--batch-size', type=int, default=8)
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--output-dir', type=str, default=None)
    args = parser.parse_args()

    # Load config
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    device = torch.device(args.device)

    # Output directory
    if args.output_dir:
        save_dir = Path(args.output_dir)
    else:
        ckpt_dir = Path(args.checkpoint).parent.parent
        save_dir = ckpt_dir / 'eval_heatmap'
    save_dir.mkdir(parents=True, exist_ok=True)

    # Build model
    logger.info("Building model...")
    model = build_model(cfg)
    model = load_checkpoint(args.checkpoint, model, device)
    model = model.to(device)
    model.eval()

    # Build dataset (with defer_heatmap_to_gpu)
    logger.info("Building dataset...")
    sw_cfg = cfg['data']['sliding_window']

    dataset = build_sliding_window_dataset(
        cfg,
        split=cfg['data'].get('val_split', 'val'),
        samples_per_clip=sw_cfg.get('val_samples_per_clip', 5),
        defer_heatmap_to_gpu=True,
    )

    packing_enabled = cfg['model']['llm'].get('enable_packing', False)
    if packing_enabled:
        raise ValueError("当前共享环境热力图评估不支持 sequence packing，请关闭 enable_packing。")
    from scripts.train import collate_fn as train_collate_fn
    collate_fn = train_collate_fn
    actual_dataset = dataset

    dataloader = DataLoader(
        actual_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
        collate_fn=collate_fn,
        prefetch_factor=2,
    )

    logger.info(f"Dataset: {len(dataset)} samples")
    logger.info("Packing: False")

    # GPU heatmap computer
    hm_size = tuple(cfg['data'].get('init_hm_size', [64, 64]))
    img_size = tuple(cfg['data'].get('image_size', [640, 480]))
    gpu_heatmap_computer = GPUHeatmapComputer(
        hm_size=hm_size,
        img_size=(img_size[0], img_size[0]),
        device=str(device),
    )

    # Run evaluation
    logger.info("=" * 60)
    logger.info("Running heatmap quality evaluation...")
    logger.info(f"  Checkpoint: {args.checkpoint}")
    logger.info(f"  Max samples: {args.max_samples}")
    logger.info(f"  Visualizations: {args.num_vis} batches")
    logger.info(f"  Output: {save_dir}")
    logger.info("=" * 60)

    results = evaluate_heatmap(
        model=model,
        dataloader=dataloader,
        gpu_heatmap_computer=gpu_heatmap_computer,
        device=device,
        save_dir=save_dir,
        max_samples=args.max_samples,
        num_vis=args.num_vis,
    )

    # Print results
    logger.info("=" * 60)
    logger.info("EVALUATION RESULTS")
    logger.info("=" * 60)
    logger.info(f"Total samples: {results['total_samples']}")
    logger.info(f"Non-empty GT: {results['nonempty_samples']} ({1 - results['empty_ratio']:.1%})")
    logger.info(f"Empty GT: {results['empty_samples']} ({results['empty_ratio']:.1%})")

    if 'nonempty' in results:
        r = results['nonempty']
        logger.info("")
        logger.info("Non-empty GT samples (model should predict peaks):")
        logger.info(f"  Peak Error:   {r['peak_error_mean']:.2f} +/- {r['peak_error_std']:.2f} px (median={r['peak_error_median']:.2f})")
        logger.info(f"  IoU@0.1:      {r['iou_0.1_mean']:.4f}")
        logger.info(f"  IoU@0.3:      {r['iou_0.3_mean']:.4f}")
        logger.info(f"  IoU@0.5:      {r['iou_0.5_mean']:.4f}")
        logger.info(f"  Cosine Sim:   {r['cosine_sim_mean']:.4f}")
        logger.info(f"  MSE:          {r['mse_mean']:.6f}")
        logger.info(f"  Pred Max:     {r['pred_max_mean']:.4f}")
        logger.info(f"  GT Max:       {r['gt_max_mean']:.4f}")

    if 'empty' in results:
        r = results['empty']
        logger.info("")
        logger.info("Empty GT samples (model should predict near-zero):")
        logger.info(f"  False Positive Energy: {r['false_positive_energy_mean']:.6f}")
        logger.info(f"  Pred Max Mean:         {r['pred_max_mean']:.4f}")
        logger.info(f"  Pred Max Max:          {r['pred_max_max']:.4f}")

    # Save results
    results_path = save_dir / 'metrics.json'
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    logger.info(f"\nMetrics saved to: {results_path}")
    logger.info(f"Visualizations saved to: {save_dir / 'visualizations'}")


if __name__ == '__main__':
    main()

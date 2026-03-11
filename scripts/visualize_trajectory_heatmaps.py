#!/usr/bin/env python3
"""
轨迹热力图可视化（HeatmapVLN v2）
=================================

沿着一条完整轨迹抽样多个位置，对每个位置聚合所有历史时间步热力图，
以 4 视角 × 4 列 (RGB | GT | Gated | Visibility 诊断) 的格式可视化。
与训练时 `visualize_heatmap_predictions` 采用完全相同的格式。
"""

import argparse
import logging
import random
import sys
from contextlib import nullcontext
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from src.data.vln_sliding_window_dataset import VLNSlidingWindowDataset
from src.models.pipeline import VLNPipeline, VLNPipelineConfig

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("vis_traj")

VIEW_LABELS = ["Front", "Right", "Back", "Left"]


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
        heatmap_lambda_coord=heatmap_cfg.get('lambda_coord', 1.0),
        heatmap_lambda_kl=heatmap_cfg.get('lambda_kl', heatmap_cfg.get('lambda_pos', 1.0)),
        heatmap_lambda_neg=heatmap_cfg.get('lambda_neg', 1.0),
        heatmap_lambda_peak=heatmap_cfg.get('lambda_peak', 1.0),
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


def infer_sample(
    model: VLNPipeline, sample: Dict[str, Any], device: torch.device,
) -> Dict[str, torch.Tensor]:
    """Run inference and return raw heatmaps, gated heatmaps, and visibility."""
    history_frames = sample['history_frames'].unsqueeze(0).to(device)
    current_frame = sample['current_frame'].unsqueeze(0).to(device)
    current_views = sample.get('current_views')
    history_panoramas = sample.get('history_panoramas')
    if current_views is None or history_panoramas is None:
        raise ValueError("数据集不包含全景 current_views/history_panoramas。")

    current_views = current_views.unsqueeze(0).to(device)
    history_panoramas = history_panoramas.unsqueeze(0).to(device)
    video_frames = torch.cat([history_frames, history_frames[:, -1:]], dim=1)
    autocast_ctx = (
        torch.autocast(device_type='cuda', dtype=torch.bfloat16)
        if device.type == 'cuda'
        else nullcontext()
    )

    with torch.no_grad(), autocast_ctx:
        outputs = model(
            video_frames=video_frames,
            instruction_text=[sample['text']],
            current_observation=current_frame,
            current_views=current_views,
            history_panoramas=history_panoramas,
            return_heatmaps=True,
            return_actions=False,
        )

    result: Dict[str, torch.Tensor] = {}
    hm = outputs.get('heatmaps')
    if hm is None:
        raise RuntimeError("模型未返回 heatmaps。")
    result['heatmaps'] = hm[0].float().cpu()           # (N_hist, 4, H, W)
    vis = outputs.get('visibility')
    if vis is not None:
        result['visibility'] = vis[0].float().cpu()     # (N_hist, 4)
    gated = outputs.get('heatmaps_gated')
    if gated is not None:
        result['heatmaps_gated'] = gated[0].float().cpu()  # (N_hist, 4, H, W)
    return result


def aggregate_max(t: torch.Tensor) -> torch.Tensor:
    """Max-aggregate over the N_hist (dim=0) dimension: (N_hist, 4, ...) → (4, ...)."""
    if t.dim() >= 3:
        return t.max(dim=0).values
    return t


def compute_gated_fallback(
    heatmaps: torch.Tensor, visibility: Optional[torch.Tensor],
) -> torch.Tensor:
    """Manually compute per-view softmax + vis gate (same logic as _gated_softmax_heatmaps)."""
    sig = heatmaps.float().clamp(1e-6, 1 - 1e-6)
    logits = torch.logit(sig)
    N_h = logits.shape[0]
    probs = torch.softmax(logits.reshape(N_h, 4, -1), dim=-1)
    probs = probs.reshape_as(heatmaps)
    if visibility is not None:
        vis_gate = torch.sigmoid(visibility.float())  # (N_h, 4)
        probs = probs * vis_gate[:, :, None, None]
    return probs


def collect_clip_samples(
    dataset: VLNSlidingWindowDataset,
    num_clips: int,
    frames_per_clip: int,
    seed: int,
) -> List[Tuple[int, List[Tuple[int, int]]]]:
    by_clip: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for dataset_idx, (clip_idx, frame_idx) in enumerate(dataset.sample_index):
        by_clip[int(clip_idx)].append((int(frame_idx), int(dataset_idx)))

    clip_items = [
        (clip_idx, sorted(items, key=lambda x: x[0]))
        for clip_idx, items in by_clip.items() if items
    ]
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


def visualize_clip_diagnostic(
    clip_name: str,
    instruction: str,
    samples_data: List[Dict[str, Any]],
    output_path: str,
) -> None:
    """
    Generate a diagnostic figure matching the training visualization format.
    Each sample occupies 4 rows (one per view direction).
    4 columns: RGB | GT heatmap | Gated heatmap | Visibility diagnostic.
    """
    num_samples = len(samples_data)
    if num_samples == 0:
        return

    total_rows = num_samples * 4
    fig, axes = plt.subplots(total_rows, 4, figsize=(16, 4 * total_rows))
    if total_rows == 1:
        axes = axes[np.newaxis, :]

    for s_idx, sd in enumerate(samples_data):
        views_np = sd['current_views']   # (4, H, W, 3)
        gt_4 = sd['gt_agg']             # (4, Hm, Wm)
        gated_4 = sd['gated_agg']       # (4, Hm, Wm)
        vis_scores = sd['vis_scores']    # (4,) float, sigmoid aggregated
        gt_vis_4 = sd['gt_vis']          # (4,) float
        n_hist = sd['n_hist']
        frame_label = sd['frame_label']
        row_offset = s_idx * 4

        for v in range(4):
            r = row_offset + v
            rgb = np.clip(views_np[v], 0, 1)
            axes[r, 0].imshow(rgb)
            label = f"F{frame_label} {VIEW_LABELS[v]}" if v == 0 else VIEW_LABELS[v]
            axes[r, 0].set_title(label, fontweight='bold')
            axes[r, 0].axis('off')

            gt_hm = gt_4[v]
            axes[r, 1].imshow(gt_hm, cmap='inferno', vmin=0, vmax=max(float(gt_hm.max()), 0.01))
            axes[r, 1].set_title(f"GT (max={float(gt_hm.max()):.2f})")
            axes[r, 1].axis('off')

            gated_v = gated_4[v]
            gated_vmax = max(float(gated_v.max()), 1e-8)
            axes[r, 2].imshow(gated_v, cmap='inferno', vmin=0, vmax=gated_vmax)
            axes[r, 2].set_title(f"Gated (max={float(gated_v.max()):.4f})")
            axes[r, 2].axis('off')

            pred_v = vis_scores[v]
            gt_v = gt_vis_4[v]
            correct = (pred_v > 0.5) == (gt_v > 0.5)
            bg_color = [0.85, 0.95, 0.85] if correct else [0.95, 0.85, 0.85]
            axes[r, 3].set_facecolor(bg_color)
            axes[r, 3].text(
                0.5, 0.55,
                f"Pred vis: {pred_v:.2f}\nGT vis: {gt_v:.0f}",
                ha='center', va='center', fontsize=14, fontfamily='monospace',
                transform=axes[r, 3].transAxes,
            )
            status = "OK" if correct else "WRONG"
            axes[r, 3].text(
                0.5, 0.15, status,
                ha='center', va='center', fontsize=16, fontweight='bold',
                color='green' if correct else 'red',
                transform=axes[r, 3].transAxes,
            )
            axes[r, 3].set_title("Visibility")
            axes[r, 3].set_xticks([])
            axes[r, 3].set_yticks([])

            if v == 0:
                axes[r, 0].set_ylabel(
                    f"Pos {s_idx}\nF{frame_label}\n(N={n_hist})",
                    fontsize=10, fontweight='bold', rotation=0,
                    labelpad=60, va='center',
                )

    title_instr = instruction[:120] + "..." if len(instruction) > 120 else instruction
    plt.suptitle(f"{clip_name} — {num_samples} positions\n{title_instr}", fontsize=12)
    plt.tight_layout(rect=[0, 0, 1, 0.98])
    fig.savefig(output_path, dpi=100, bbox_inches='tight')
    plt.close(fig)
    logger.info(f"  Saved: {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="轨迹热力图可视化 (HeatmapVLN v2)")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num-clips', type=int, default=3, help='可视化 clip 数')
    parser.add_argument('--frames-per-clip', type=int, default=6, help='每个 clip 抽样的位置数')
    parser.add_argument('--output-dir', type=str, default='./vis_trajectory')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', type=str, default=None, help='默认使用配置里的 val_split')
    parser.add_argument('--attn-impl', type=str, default=None,
                        help='覆盖 attention implementation (sdpa/flash_attention_2)')
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

    if args.attn_impl:
        cfg['model']['llm']['attn_implementation'] = args.attn_impl
        logger.info(f"  Overriding attn_implementation → {args.attn_impl}")

    logger.info("Building model...")
    model = build_model(cfg, device=str(device))
    model = model.to(device)
    model._ensure_heatmap_vln()
    state_dict = ckpt.get('trainable_state_dict', ckpt.get('model_state_dict', {}))
    if state_dict:
        current_state = model.state_dict()
        norm = lambda n: n.replace("module.", "", 1).replace(".module.", ".")
        norm_to_actual = {norm(k): k for k in current_state.keys()}
        remapped = {
            norm_to_actual[norm(k)]: v
            for k, v in state_dict.items() if norm(k) in norm_to_actual
        }
        missing, unexpected = model.load_state_dict(remapped, strict=False)
        logger.info(
            f"  Loaded: {len(remapped)}/{len(state_dict)} params, "
            f"missing={len(missing)}, unexpected={len(unexpected)}"
        )
    else:
        logger.warning("  No trainable weights in checkpoint!")
    model.eval()

    logger.info(f"Visualizing {len(selected_clips)} clips...")
    for clip_order, (clip_idx, items) in enumerate(selected_clips):
        clip_dir = dataset.clips[clip_idx]
        clip_name = f"{clip_dir.parent.name}/{clip_dir.name}"
        logger.info(f"[Clip {clip_order + 1}/{len(selected_clips)}] {clip_name} ({len(items)} samples)")

        samples_data: List[Dict[str, Any]] = []
        instruction = ""

        for frame_idx, dataset_idx in items:
            sample = dataset[dataset_idx]
            instruction = sample['text']

            result = infer_sample(model, sample, device=device)
            pred_raw = result['heatmaps']           # (N_hist, 4, H, W) raw sigmoid
            pred_vis = result.get('visibility')     # (N_hist, 4) logits or None
            pred_gated = result.get('heatmaps_gated')  # (N_hist, 4, H, W) or None

            gt_all = sample['heatmap'].float().cpu()  # (N_hist, 4, H, W)
            gt_vis_raw = sample.get('gt_visibility')  # (N_hist, 4) or None

            n_hist = gt_all.shape[0] if gt_all.dim() == 4 else 1

            # --- GT: max-aggregate across N_hist → (4, H, W) ---
            gt_agg = aggregate_max(gt_all).numpy()

            # --- Gated pred: max-aggregate across N_hist → (4, H, W) ---
            if pred_gated is not None:
                gated_agg = aggregate_max(pred_gated).numpy()
            else:
                gated_t = compute_gated_fallback(pred_raw, pred_vis)
                gated_agg = aggregate_max(gated_t).numpy()

            # --- Visibility scores: sigmoid → max-aggregate across N_hist → (4,) ---
            if pred_vis is not None:
                vis_sig = torch.sigmoid(pred_vis)  # (N_hist, 4)
                vis_scores = vis_sig.max(dim=0).values.numpy()  # (4,)
            else:
                vis_scores = np.ones(4)

            # --- GT visibility: max-aggregate → (4,) ---
            if gt_vis_raw is not None:
                if gt_vis_raw.dim() == 2:
                    gt_vis_4 = gt_vis_raw.float().max(dim=0).values.numpy()
                else:
                    gt_vis_4 = gt_vis_raw.float().numpy()
            else:
                gt_vis_4 = (gt_agg.max(axis=(-2, -1)) > 0).astype(float)

            views_np = sample['current_views'].cpu().numpy().transpose(0, 2, 3, 1)  # (4, H, W, 3)

            gated_has_signal = "gated" if pred_gated is not None else "fallback"
            logger.info(
                "  frame=%4d N_hist=%d src=%s  gt_max=[%.2f, %.2f, %.2f, %.2f]  "
                "gated_max=[%.4f, %.4f, %.4f, %.4f]  vis=[%.2f, %.2f, %.2f, %.2f]",
                frame_idx, n_hist, gated_has_signal,
                *[float(gt_agg[v].max()) for v in range(4)],
                *[float(gated_agg[v].max()) for v in range(4)],
                *[float(vis_scores[v]) for v in range(4)],
            )

            samples_data.append({
                'current_views': views_np,
                'gt_agg': gt_agg,
                'gated_agg': gated_agg,
                'vis_scores': vis_scores,
                'gt_vis': gt_vis_4,
                'n_hist': n_hist,
                'frame_label': frame_idx,
            })

        out_path = output_dir / f"clip_{clip_order:02d}_{clip_dir.name}.png"
        visualize_clip_diagnostic(
            clip_name=clip_name,
            instruction=instruction,
            samples_data=samples_data,
            output_path=str(out_path),
        )
        torch.cuda.empty_cache()

    logger.info(f"All done! Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

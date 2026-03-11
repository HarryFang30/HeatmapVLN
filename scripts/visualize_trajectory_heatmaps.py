#!/usr/bin/env python3
"""
轨迹热力图可视化（HeatmapVLN v2 — 全景横向布局 + BEV）
=======================================================

每个位置的 4 视角 (Front|Right|Back|Left) 在同一行并排，
所有位置水平排列，顶部显示 BEV 俯视轨迹图。
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
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
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

VIEW_LABELS = ["F", "R", "B", "L"]
TILE = 80
GAP = 4
SEP_COLOR = np.array([1.0, 0.85, 0.0], dtype=np.float32)  # yellow separator


# ───────────────────── Model ─────────────────────

def build_model(cfg: Dict, device: str) -> VLNPipeline:
    model_cfg = cfg['model']
    llm_cfg = model_cfg.get('llm', {})
    heatmap_cfg = model_cfg.get('heatmap', {})
    action_cfg = model_cfg.get('action_head', {})
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
        action_head_type=action_cfg.get('type', 'transformer'),
        enable_action_head=False,
        enable_stop_head=False,
        enable_progress_head=False,
        verbose=False,
    )
    return VLNPipeline(config)


# ───────────────────── Inference ─────────────────────

def infer_sample(
    model: VLNPipeline, sample: Dict[str, Any], device: torch.device,
) -> Dict[str, torch.Tensor]:
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
        if device.type == 'cuda' else nullcontext()
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
    result['heatmaps'] = hm[0].float().cpu()
    vis = outputs.get('visibility')
    if vis is not None:
        result['visibility'] = vis[0].float().cpu()
    gated = outputs.get('heatmaps_gated')
    if gated is not None:
        result['heatmaps_gated'] = gated[0].float().cpu()
    return result


def aggregate_max(t: torch.Tensor) -> torch.Tensor:
    if t.dim() >= 3:
        return t.max(dim=0).values
    return t


def compute_gated_fallback(
    heatmaps: torch.Tensor, visibility: Optional[torch.Tensor],
) -> torch.Tensor:
    sig = heatmaps.float().clamp(1e-6, 1 - 1e-6)
    logits = torch.logit(sig)
    N_h = logits.shape[0]
    probs = torch.softmax(logits.reshape(N_h, 4, -1), dim=-1)
    probs = probs.reshape_as(heatmaps)
    if visibility is not None:
        vis_gate = torch.sigmoid(visibility.float())
        probs = probs * vis_gate[:, :, None, None]
    return probs


# ───────────────────── Strip builders ─────────────────────

def _resize(img: np.ndarray, size: int) -> np.ndarray:
    if img.shape[0] == size and img.shape[1] == size:
        return img
    return cv2.resize(img.astype(np.float32), (size, size), interpolation=cv2.INTER_AREA)


def _fill_sep(strip: np.ndarray, x_start: int, gap: int) -> None:
    """Fill a gap region with the separator color."""
    strip[:, x_start: x_start + gap] = SEP_COLOR


def _build_rgb_strip(samples_data: List[Dict], tile: int = TILE, gap: int = GAP) -> np.ndarray:
    N = len(samples_data)
    group_w = 4 * tile
    strip_w = N * group_w + max(N - 1, 0) * gap
    strip = np.zeros((tile, strip_w, 3), dtype=np.float32)
    for i, sd in enumerate(samples_data):
        views = sd['current_views']
        x0 = i * (group_w + gap)
        for v in range(4):
            t = _resize(np.clip(views[v], 0, 1), tile)
            strip[:, x0 + v * tile: x0 + (v + 1) * tile] = t
        if i < N - 1:
            _fill_sep(strip, x0 + group_w, gap)
    return strip


def _build_heatmap_strip(
    samples_data: List[Dict], key: str,
    tile: int = TILE, gap: int = GAP,
    global_vmax: Optional[float] = None,
) -> np.ndarray:
    cmap = plt.cm.inferno
    N = len(samples_data)
    group_w = 4 * tile
    strip_w = N * group_w + max(N - 1, 0) * gap
    strip = np.zeros((tile, strip_w, 3), dtype=np.float32)
    for i, sd in enumerate(samples_data):
        hm_4 = sd[key]  # (4, Hm, Wm)
        if global_vmax is not None:
            vmax = global_vmax
        else:
            vmax = max(float(hm_4.max()), 1e-8)
        x0 = i * (group_w + gap)
        for v in range(4):
            hm = _resize(hm_4[v], tile)
            hm_norm = np.clip(hm / vmax, 0, 1)
            t = cmap(hm_norm)[:, :, :3].astype(np.float32)
            strip[:, x0 + v * tile: x0 + (v + 1) * tile] = t
        if i < N - 1:
            _fill_sep(strip, x0 + group_w, gap)
    return strip


# ───────────────────── BEV plot ─────────────────────

def _load_topdown_data(clip_dir: Path) -> Tuple[Optional[np.ndarray], Optional[List]]:
    """Load pre-rendered navmesh BEV image and trajectory pixel coordinates."""
    topdown_path = clip_dir / "topdown_trajectory.jpg"
    transform_path = clip_dir / "topdown_transform.json"

    topdown_img = cv2.imread(str(topdown_path))
    if topdown_img is None:
        return None, None
    topdown_img = cv2.cvtColor(topdown_img, cv2.COLOR_BGR2RGB)

    if transform_path.exists():
        import json
        with open(transform_path, 'r') as f:
            transform = json.load(f)
        trajectory_pixels = transform["trajectory_pixels"]
    else:
        trajectory_pixels = None
    return topdown_img, trajectory_pixels


def _plot_bev(
    ax: plt.Axes,
    clip_dir: Path,
    sampled_frame_indices: List[int],
    samples_data: List[Dict],
    all_poses: List[np.ndarray],
) -> None:
    topdown_img, trajectory_pixels = _load_topdown_data(clip_dir)

    if topdown_img is not None and trajectory_pixels is not None:
        canvas = topdown_img.copy()
        N = len(sampled_frame_indices)
        colors_rgb = [
            tuple(int(c * 255) for c in plt.cm.tab20(i / max(N - 1, 1))[:3])
            for i in range(N)
        ]

        num_pts = len(trajectory_pixels)
        for i in range(num_pts - 1):
            p1 = tuple(trajectory_pixels[i])
            p2 = tuple(trajectory_pixels[i + 1])
            cv2.line(canvas, p1, p2, (200, 200, 200), 1, cv2.LINE_AA)

        for i, fidx in enumerate(sampled_frame_indices):
            if fidx >= num_pts:
                continue
            curr = tuple(trajectory_pixels[fidx])
            bgr = (colors_rgb[i][2], colors_rgb[i][1], colors_rgb[i][0])
            cv2.circle(canvas, curr, 6, bgr, -1, cv2.LINE_AA)
            cv2.circle(canvas, curr, 6, (0, 0, 0), 1, cv2.LINE_AA)

            if fidx + 1 < num_pts:
                nxt = tuple(trajectory_pixels[fidx + 1])
                dx, dy = nxt[0] - curr[0], nxt[1] - curr[1]
                length = max(1, (dx ** 2 + dy ** 2) ** 0.5)
                arrow_len = 18
                tip = (int(curr[0] + dx / length * arrow_len),
                       int(curr[1] + dy / length * arrow_len))
                cv2.arrowedLine(canvas, curr, tip, bgr, 2, tipLength=0.35)

            label_pos = (curr[0] + 8, curr[1] - 4)
            cv2.putText(canvas, str(i), label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 0), 2, cv2.LINE_AA)
            cv2.putText(canvas, str(i), label_pos,
                        cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        ax.imshow(canvas)
        ax.axis('off')
    else:
        positions = np.array([p[:3, 3] for p in all_poses])
        x_all, z_all = positions[:, 0], positions[:, 2]
        ax.plot(x_all, z_all, '-', color='#bbbbbb', linewidth=1.2, zorder=1)

        N = len(sampled_frame_indices)
        colors = plt.cm.tab20(np.linspace(0, 1, max(N, 1)))
        extent = max(x_all.max() - x_all.min(), z_all.max() - z_all.min(), 1e-3)
        arrow_len = extent * 0.04

        for i, fidx in enumerate(sampled_frame_indices):
            px, pz = x_all[fidx], z_all[fidx]
            ax.scatter(px, pz, c=[colors[i]], s=40, marker='o', zorder=5,
                       edgecolors='k', linewidth=0.5)
            ax.annotate(str(i), (px, pz), fontsize=5, ha='center', va='bottom',
                        xytext=(0, 4), textcoords='offset points', fontweight='bold')
            fwd = -all_poses[fidx][:3, 2]
            fwd_bev = np.array([fwd[0], fwd[2]])
            fwd_norm = np.linalg.norm(fwd_bev)
            if fwd_norm > 1e-6:
                fwd_bev = fwd_bev / fwd_norm * arrow_len
                ax.annotate('', xy=(px + fwd_bev[0], pz + fwd_bev[1]), xytext=(px, pz),
                            arrowprops=dict(arrowstyle='->', color=colors[i], lw=1.2))
        ax.set_aspect('equal')
        ax.tick_params(labelsize=6)
        ax.grid(True, alpha=0.2, linewidth=0.5)


# ───────────────────── Main visualization ─────────────────────

def visualize_clip_panoramic(
    clip_name: str,
    instruction: str,
    samples_data: List[Dict[str, Any]],
    all_poses: List[np.ndarray],
    sampled_frame_indices: List[int],
    clip_dir: Path,
    output_path: str,
    tile: int = TILE,
    gap: int = GAP,
) -> None:
    N = len(samples_data)
    if N == 0:
        return

    rgb_strip = _build_rgb_strip(samples_data, tile, gap)
    gt_strip = _build_heatmap_strip(samples_data, 'gt_agg', tile, gap, global_vmax=1.0)
    gated_strip = _build_heatmap_strip(samples_data, 'gated_agg', tile, gap, global_vmax=None)

    strip_w_px = rgb_strip.shape[1]
    strip_h_px = tile
    group_w = 4 * tile

    DPI = 120
    bev_h_ratio = 3
    fig_w_in = strip_w_px / DPI
    strip_h_in = strip_h_px / DPI
    bev_h_in = strip_h_in * bev_h_ratio
    total_h_in = bev_h_in + strip_h_in * 3
    fig_w_in = max(fig_w_in, 10)
    total_h_in = max(total_h_in, 3)

    fig = plt.figure(figsize=(fig_w_in, total_h_in))
    gs = fig.add_gridspec(
        4, 1, height_ratios=[bev_h_ratio, 1, 1, 1],
        hspace=0.05, left=0.02, right=0.98, top=0.95, bottom=0.02,
    )

    ax_bev = fig.add_subplot(gs[0])
    _plot_bev(ax_bev, clip_dir, sampled_frame_indices, samples_data, all_poses)
    instr_short = instruction[:120] + "..." if len(instruction) > 120 else instruction
    ax_bev.set_title(f"BEV: {clip_name}\n{instr_short}", fontsize=7, pad=4)

    row_data = [
        (gs[1], rgb_strip, "RGB (F|R|B|L)"),
        (gs[2], gt_strip, "GT Heatmap"),
        (gs[3], gated_strip, "Gated"),
    ]
    for gs_slot, strip, ylabel in row_data:
        ax = fig.add_subplot(gs_slot)
        ax.imshow(strip, aspect='equal', interpolation='nearest')
        ax.set_ylabel(ylabel, fontsize=7, rotation=90, labelpad=8)
        ax.set_yticks([])

        tick_xs = [i * (group_w + gap) + group_w // 2 for i in range(N)]
        tick_labels = [str(i) for i in range(N)]
        ax.set_xticks(tick_xs)
        ax.set_xticklabels(tick_labels, fontsize=5)

    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    logger.info(f"  Saved: {output_path}")


# ───────────────────── Data collection ─────────────────────

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


# ───────────────────── Main ─────────────────────

def main() -> int:
    parser = argparse.ArgumentParser(description="轨迹热力图可视化 (HeatmapVLN v2 — 全景横向 + BEV)")
    parser.add_argument('--checkpoint', type=str, required=True)
    parser.add_argument('--num-clips', type=int, default=3)
    parser.add_argument('--frames-per-clip', type=int, default=32)
    parser.add_argument('--output-dir', type=str, default='./vis_trajectory')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--attn-impl', type=str, default=None)
    parser.add_argument('--tile-size', type=int, default=TILE)
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
        logger.error("数据集不是全景 4 视角格式。")
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

        all_poses = dataset._load_poses(clip_idx)
        samples_data: List[Dict[str, Any]] = []
        sampled_frame_indices: List[int] = []
        instruction = ""

        for frame_idx, dataset_idx in items:
            sample = dataset[dataset_idx]
            instruction = sample['text']
            sampled_frame_indices.append(frame_idx)

            result = infer_sample(model, sample, device=device)
            pred_raw = result['heatmaps']
            pred_vis = result.get('visibility')
            pred_gated = result.get('heatmaps_gated')

            gt_all = sample['heatmap'].float().cpu()
            gt_vis_raw = sample.get('gt_visibility')

            n_hist = gt_all.shape[0] if gt_all.dim() == 4 else 1
            gt_agg = aggregate_max(gt_all).numpy()

            if pred_gated is not None:
                gated_agg = aggregate_max(pred_gated).numpy()
            else:
                gated_agg = aggregate_max(compute_gated_fallback(pred_raw, pred_vis)).numpy()

            if pred_vis is not None:
                vis_scores = torch.sigmoid(pred_vis).max(dim=0).values.numpy()
            else:
                vis_scores = np.ones(4)

            if gt_vis_raw is not None:
                gt_vis_4 = gt_vis_raw.float().max(dim=0).values.numpy() if gt_vis_raw.dim() == 2 else gt_vis_raw.float().numpy()
            else:
                gt_vis_4 = (gt_agg.max(axis=(-2, -1)) > 0).astype(float)

            views_np = sample['current_views'].cpu().numpy().transpose(0, 2, 3, 1)

            logger.info(
                "  frame=%4d N=%d  gt_max=[%.2f,%.2f,%.2f,%.2f]  "
                "gated_max=[%.4f,%.4f,%.4f,%.4f]  vis=[%.2f,%.2f,%.2f,%.2f]",
                frame_idx, n_hist,
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
        visualize_clip_panoramic(
            clip_name=clip_name,
            instruction=instruction,
            samples_data=samples_data,
            all_poses=all_poses,
            sampled_frame_indices=sampled_frame_indices,
            clip_dir=clip_dir,
            output_path=str(out_path),
            tile=args.tile_size,
        )
        torch.cuda.empty_cache()

    logger.info(f"All done! Output: {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

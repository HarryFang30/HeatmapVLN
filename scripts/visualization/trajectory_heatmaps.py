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
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from scripts.training.model_builder import build_model
from scripts.training.utils import make_autocast_context, load_checkpoint

from src.data.factory import build_sliding_window_dataset
from src.data.trajectory_utils import get_trajectory_relative_to_frame
from src.data.vln_sliding_window_dataset import VLNSlidingWindowDataset
from src.models.pipeline import VLNPipeline

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%H:%M:%S')
logger = logging.getLogger("vis_traj")

VIEW_LABELS = ["F", "R", "B", "L"]
TILE = 80
GAP = 4
SEP_COLOR = np.array([1.0, 0.85, 0.0], dtype=np.float32)  # yellow separator


# ───────────────────── Inference ─────────────────────

def infer_sample(
    model: VLNPipeline,
    sample: dict[str, Any],
    device: torch.device,
    amp_type: str = "bf16",
) -> dict[str, torch.Tensor]:
    history_frames = sample['history_frames'].unsqueeze(0).to(device)
    current_frame = sample['current_frame'].unsqueeze(0).to(device)
    current_views = sample.get('current_views')
    history_panoramas = sample.get('history_panoramas')
    if current_views is None or history_panoramas is None:
        raise ValueError("数据集不包含全景 current_views/history_panoramas。")

    current_views = current_views.unsqueeze(0).to(device)
    history_panoramas = history_panoramas.unsqueeze(0).to(device)
    video_frames = torch.cat([history_frames, history_frames[:, -1:]], dim=1)
    with torch.no_grad(), make_autocast_context(device, amp_type):
        outputs = model(
            video_frames=video_frames,
            instruction_text=[sample['text']],
            current_observation=current_frame,
            current_views=current_views,
            history_panoramas=history_panoramas,
            return_heatmaps=True,
            return_actions=False,
        )

    result: dict[str, torch.Tensor] = {}
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
    heatmaps: torch.Tensor, visibility: torch.Tensor | None,
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


def _build_rgb_strip(samples_data: list[dict], tile: int = TILE, gap: int = GAP) -> np.ndarray:
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
    samples_data: list[dict], key: str,
    tile: int = TILE, gap: int = GAP,
    global_vmax: float | None = None,
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
            if float(hm.max()) < 1e-8:
                strip[:, x0 + v * tile: x0 + (v + 1) * tile] = 0.0
            else:
                hm_norm = np.clip(hm / vmax, 0, 1)
                t = cmap(hm_norm)[:, :, :3].astype(np.float32)
                strip[:, x0 + v * tile: x0 + (v + 1) * tile] = t
        if i < N - 1:
            _fill_sep(strip, x0 + group_w, gap)
    return strip


# ───────────────────── Per-position top-down strip ─────────────────────

def _load_topdown_data(clip_dir: Path) -> tuple[np.ndarray | None, list | None]:
    """Load pre-rendered navmesh BEV image and trajectory pixel coordinates."""
    topdown_path = clip_dir / "topdown_trajectory.jpg"
    transform_path = clip_dir / "topdown_transform.json"

    topdown_img = cv2.imread(str(topdown_path))
    if topdown_img is None:
        return None, None
    topdown_img = cv2.cvtColor(topdown_img, cv2.COLOR_BGR2RGB)

    if transform_path.exists():
        import json
        with open(transform_path) as f:
            transform = json.load(f)
        trajectory_pixels = transform["trajectory_pixels"]
    else:
        trajectory_pixels = None
    return topdown_img, trajectory_pixels


def _heading_from_pose(pose: np.ndarray) -> tuple[float, float]:
    """Extract the agent's forward direction on the XZ ground plane from a
    camera-to-world 4x4 pose matrix.  Camera forward is -Z in camera space,
    so world forward = R @ [0, 0, -1]^T.

    Returns (fwd_x, fwd_z) normalised direction in world coordinates.
    """
    R = pose[:3, :3]
    fwd = R @ np.array([0.0, 0.0, -1.0])
    fx, fz = float(fwd[0]), float(fwd[2])
    length = max(np.hypot(fx, fz), 1e-8)
    return fx / length, fz / length


def _render_topdown_for_position(
    base_img: np.ndarray,
    trajectory_pixels: list,
    current_frame_idx: int,
    history_frame_indices: list[int],
    position_label: int,
    out_size: int,
    current_pose: np.ndarray | None = None,
) -> np.ndarray:
    """Render a top-down map highlighting current position and its history frames."""
    canvas = base_img.copy()
    num_pts = len(trajectory_pixels)

    for i in range(num_pts - 1):
        p1 = tuple(trajectory_pixels[i])
        p2 = tuple(trajectory_pixels[i + 1])
        cv2.line(canvas, p1, p2, (200, 200, 200), 1, cv2.LINE_AA)

    HIST_COLORS = [
        (50, 200, 50), (0, 180, 180), (200, 180, 0), (180, 100, 255),
        (0, 130, 255), (255, 130, 0), (130, 255, 0), (200, 0, 200),
    ]

    for h_idx, h_frame in enumerate(history_frame_indices):
        if h_frame >= num_pts:
            continue
        pt = tuple(trajectory_pixels[h_frame])
        color = HIST_COLORS[h_idx % len(HIST_COLORS)]
        cv2.circle(canvas, pt, 7, color, -1, cv2.LINE_AA)
        cv2.circle(canvas, pt, 7, (0, 0, 0), 1, cv2.LINE_AA)
        lbl = f"h{h_idx}"
        cv2.putText(canvas, lbl, (pt[0] + 9, pt[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, (0, 0, 0), 2, cv2.LINE_AA)
        cv2.putText(canvas, lbl, (pt[0] + 9, pt[1] + 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.38, color, 1, cv2.LINE_AA)

    if current_frame_idx < num_pts:
        curr = tuple(trajectory_pixels[current_frame_idx])
        cv2.circle(canvas, curr, 12, (255, 0, 0), 3, cv2.LINE_AA)
        cv2.circle(canvas, curr, 5, (255, 0, 0), -1, cv2.LINE_AA)

        if current_pose is not None:
            fwd_x, fwd_z = _heading_from_pose(current_pose)
            arrow_len = 30
            tip = (int(curr[0] + fwd_x * arrow_len),
                   int(curr[1] + fwd_z * arrow_len))
            cv2.arrowedLine(canvas, curr, tip, (255, 0, 0), 2, tipLength=0.3)
        elif current_frame_idx + 1 < num_pts:
            nxt = tuple(trajectory_pixels[current_frame_idx + 1])
            dx, dy = nxt[0] - curr[0], nxt[1] - curr[1]
            length = max(1, (dx ** 2 + dy ** 2) ** 0.5)
            arrow_len = 25
            tip = (int(curr[0] + dx / length * arrow_len),
                   int(curr[1] + dy / length * arrow_len))
            cv2.arrowedLine(canvas, curr, tip, (255, 0, 0), 2, tipLength=0.3)

        label = str(position_label)
        cv2.putText(canvas, label, (curr[0] + 14, curr[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 3, cv2.LINE_AA)
        cv2.putText(canvas, label, (curr[0] + 14, curr[1] - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 50, 50), 2, cv2.LINE_AA)

    canvas = cv2.resize(canvas, (out_size, out_size), interpolation=cv2.INTER_AREA)
    return canvas.astype(np.float32) / 255.0


def _build_topdown_strip(
    clip_dir: Path,
    sampled_frame_indices: list[int],
    samples_data: list[dict[str, Any]],
    all_poses: list[np.ndarray] | None = None,
    tile: int = TILE, gap: int = GAP,
) -> np.ndarray | None:
    topdown_img, trajectory_pixels = _load_topdown_data(clip_dir)
    if topdown_img is None or trajectory_pixels is None:
        return None

    N = len(sampled_frame_indices)
    td_size = 4 * tile
    strip_w = N * td_size + max(N - 1, 0) * gap
    strip = np.zeros((td_size, strip_w, 3), dtype=np.float32)

    for i in range(N):
        fi = sampled_frame_indices[i]
        hist_indices = samples_data[i].get('history_indices', [])
        cur_pose = all_poses[fi] if (all_poses is not None and fi < len(all_poses)) else None
        td = _render_topdown_for_position(
            topdown_img, trajectory_pixels,
            current_frame_idx=fi,
            history_frame_indices=hist_indices,
            position_label=i,
            out_size=td_size,
            current_pose=cur_pose,
        )
        x0 = i * (td_size + gap)
        strip[:, x0: x0 + td_size] = td
        if i < N - 1:
            _fill_sep(strip, x0 + td_size, gap)

    return strip


# ───────────────── Local-frame trajectory BEV ─────────────────

def _compute_local_history(
    all_poses: list[np.ndarray],
    current_t: int,
    history_indices: list[int],
) -> np.ndarray:
    """Use the same coordinate transform as the dataloader to compute
    history trajectory in the current frame's local coordinate system,
    then convert to intuitive BEV coordinates (x=right, y=forward).

    get_trajectory_relative_to_frame returns (backward, left) convention;
    we negate and swap to (right, forward) for visualization.

    Returns:
        history_xy:  (N_hist, 2) — BEV coords (right, forward)
    """
    if len(history_indices) == 0:
        return np.zeros((0, 2), dtype=np.float32)

    hist_poses = np.stack(
        [all_poses[current_t]] + [all_poses[i] for i in history_indices],
        axis=0,
    )
    hist_rel = get_trajectory_relative_to_frame(hist_poses, camera_deg=0)
    raw = hist_rel[1:, :2]  # (backward, left)
    return np.column_stack([-raw[:, 1], -raw[:, 0]])  # (right, forward)


def _render_local_traj_bev(
    history_xy: np.ndarray,
    out_size: int,
    position_label: int,
    margin_m: float = 0.5,
) -> np.ndarray:
    """Render a small BEV image of the history trajectory in current
    frame's local coordinate system.

    Input convention (after _compute_local_history conversion):
      x = right, y = forward.  We draw y upward on the image.
    """
    canvas = np.ones((out_size, out_size, 3), dtype=np.uint8) * 245

    all_pts = np.concatenate([history_xy, np.zeros((1, 2))], axis=0)
    if len(all_pts) < 2:
        return canvas.astype(np.float32) / 255.0

    xmin, ymin = all_pts.min(axis=0) - margin_m
    xmax, ymax = all_pts.max(axis=0) + margin_m
    span = max(xmax - xmin, ymax - ymin, 0.5)

    def to_px(xy):
        px = int((xy[0] - xmin) / span * (out_size - 20) + 10)
        py = int((1.0 - (xy[1] - ymin) / span) * (out_size - 20) + 10)
        return (px, py)

    ox, oy = to_px(np.zeros(2))
    cv2.line(canvas, (ox, 0), (ox, out_size), (210, 210, 210), 1)
    cv2.line(canvas, (0, oy), (out_size, oy), (210, 210, 210), 1)

    # history trail
    if len(history_xy) > 1:
        for i in range(len(history_xy) - 1):
            cv2.line(canvas, to_px(history_xy[i]), to_px(history_xy[i + 1]),
                     (200, 130, 50), 2, cv2.LINE_AA)
    for _idx, pt in enumerate(history_xy):
        cv2.circle(canvas, to_px(pt), 4, (200, 130, 50), -1, cv2.LINE_AA)

    # current position (red, at origin)
    cv2.circle(canvas, to_px(np.zeros(2)), 7, (0, 0, 255), -1, cv2.LINE_AA)
    cv2.circle(canvas, to_px(np.zeros(2)), 7, (0, 0, 0), 1, cv2.LINE_AA)

    # forward arrow (y+ direction)
    arrow_m = span * 0.12
    tip_px = to_px(np.array([0.0, arrow_m]))
    cv2.arrowedLine(canvas, to_px(np.zeros(2)), tip_px,
                    (0, 0, 255), 2, cv2.LINE_AA, tipLength=0.35)

    cv2.putText(canvas, str(position_label), (6, 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (80, 80, 80), 1, cv2.LINE_AA)

    return canvas.astype(np.float32) / 255.0


def _build_local_traj_strip(
    all_poses: list[np.ndarray],
    sampled_frame_indices: list[int],
    samples_data: list[dict[str, Any]],
    tile: int = TILE, gap: int = GAP,
) -> np.ndarray:
    N = len(sampled_frame_indices)
    cell_size = 4 * tile
    strip_w = N * cell_size + max(N - 1, 0) * gap
    strip = np.zeros((cell_size, strip_w, 3), dtype=np.float32)

    for i in range(N):
        current_t = sampled_frame_indices[i]
        hist_indices = samples_data[i].get('history_indices', [])
        history_xy = _compute_local_history(all_poses, current_t, hist_indices)
        cell = _render_local_traj_bev(
            history_xy, out_size=cell_size, position_label=i,
        )
        x0 = i * (cell_size + gap)
        strip[:, x0: x0 + cell_size] = cell
        if i < N - 1:
            _fill_sep(strip, x0 + cell_size, gap)

    return strip


# ───────────────────── Main visualization ─────────────────────

def visualize_clip_panoramic(
    clip_name: str,
    instruction: str,
    samples_data: list[dict[str, Any]],
    all_poses: list[np.ndarray],
    sampled_frame_indices: list[int],
    clip_dir: Path,
    output_path: str,
    tile: int = TILE,
    gap: int = GAP,
) -> None:
    N = len(samples_data)
    if N == 0:
        return

    td_strip = _build_topdown_strip(clip_dir, sampled_frame_indices, samples_data, all_poses, tile, gap)
    local_traj_strip = _build_local_traj_strip(
        all_poses, sampled_frame_indices, samples_data, tile, gap,
    )
    rgb_strip = _build_rgb_strip(samples_data, tile, gap)
    gt_strip = _build_heatmap_strip(samples_data, 'gt_agg', tile, gap, global_vmax=1.0)
    gated_strip = _build_heatmap_strip(samples_data, 'gated_agg', tile, gap, global_vmax=None)

    group_w = 4 * tile
    td_h = 4 * tile  # top-down tile height = group width (square)

    strips: list[tuple[np.ndarray, str]] = []
    if td_strip is not None:
        strips.append((td_strip, "Top-Down"))
    strips.append((local_traj_strip, "Local Traj"))
    strips.append((rgb_strip, "RGB (F|R|B|L)"))
    strips.append((gt_strip, "GT Heatmap"))
    strips.append((gated_strip, "Gated"))

    DPI = 120
    strip_w_px = rgb_strip.shape[1]
    fig_w_in = max(strip_w_px / DPI, 10)

    total_h_px = sum(s.shape[0] for s, _ in strips)
    fig_h_in = max(total_h_px / DPI, 3)

    height_ratios = [s.shape[0] for s, _ in strips]

    fig = plt.figure(figsize=(fig_w_in, fig_h_in))
    gs = fig.add_gridspec(
        len(strips), 1, height_ratios=height_ratios,
        hspace=0.03, left=0.02, right=0.98, top=0.97, bottom=0.01,
    )

    instr_short = instruction[:140] + "..." if len(instruction) > 140 else instruction
    fig.suptitle(f"{clip_name}  |  {instr_short}", fontsize=7, y=0.995)

    for row_idx, (strip, ylabel) in enumerate(strips):
        ax = fig.add_subplot(gs[row_idx])
        ax.imshow(strip, aspect='equal', interpolation='nearest')
        ax.set_ylabel(ylabel, fontsize=6, rotation=90, labelpad=8)
        ax.set_yticks([])

        cw = group_w if strip.shape[0] == tile else td_h
        tick_xs = [i * (cw + gap) + cw // 2 for i in range(N)]
        tick_labels = [str(i) for i in range(N)]
        ax.set_xticks(tick_xs)
        ax.set_xticklabels(tick_labels, fontsize=4)

    fig.savefig(output_path, dpi=DPI, bbox_inches='tight', pad_inches=0.08)
    plt.close(fig)
    logger.info(f"  Saved: {output_path}")


# ───────────────────── Data collection ─────────────────────

def collect_clip_samples(
    dataset: VLNSlidingWindowDataset,
    num_clips: int,
    frames_per_clip: int,
    seed: int,
) -> list[tuple[int, list[tuple[int, int]]]]:
    by_clip: dict[int, list[tuple[int, int]]] = defaultdict(list)
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
    parser.add_argument('--data-root', type=str, default=None,
                        help='Override data root (default: use checkpoint config)')
    parser.add_argument('--num-clips', type=int, default=3)
    parser.add_argument('--frames-per-clip', type=int, default=32)
    parser.add_argument('--output-dir', type=str, default='./vis_trajectory')
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--split', type=str, default=None)
    parser.add_argument('--attn-impl', type=str, default=None)
    parser.add_argument('--tile-size', type=int, default=TILE)
    parser.add_argument('--vis-threshold', type=float, default=None,
                        help='Hard visibility threshold: views with pred_vis < threshold are zeroed out')
    parser.add_argument('--peak-ratio', type=float, default=None,
                        help='Filter diffuse heatmaps: zero out views where max/mean < this ratio')
    args = parser.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else 'cpu')
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Loading checkpoint: {args.checkpoint}")
    ckpt = load_checkpoint(args.checkpoint, map_location='cpu', trust_checkpoint=True)
    cfg = ckpt['config']
    logger.info(f"  Epoch: {ckpt.get('epoch', '?')}")

    data_root = args.data_root or cfg['data']['root']
    split = args.split or cfg['data'].get('val_split', 'val')
    sw_cfg = cfg['data']['sliding_window']

    logger.info("Loading dataset...")
    dataset = build_sliding_window_dataset(
        cfg, split=split,
        root=data_root,
        clip_level_sampling=False,
        samples_per_clip=sw_cfg.get('val_samples_per_clip', 2),
        enable_augmentation=False,
        defer_heatmap_to_gpu=False,
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
    model = build_model(cfg, device=str(device), verbose=False, enable_action_head=False)
    model = model.to(device)
    model._ensure_heatmap_vln()
    state_dict = ckpt.get('trainable_state_dict', ckpt.get('model_state_dict', {}))
    if state_dict:
        current_state = model.state_dict()
        def norm(n):
            return n.replace("module.", "", 1).replace(".module.", ".")
        norm_to_actual = {norm(k): k for k in current_state}
        remapped = {}
        skipped = []
        for k, v in state_dict.items():
            actual = norm_to_actual.get(norm(k))
            if actual is not None:
                if current_state[actual].shape != v.shape:
                    skipped.append(f"{actual}: ckpt {tuple(v.shape)} vs model {tuple(current_state[actual].shape)}")
                    continue
                remapped[actual] = v
        if skipped:
            logger.warning("Skipped %d params (shape mismatch):\n  %s", len(skipped), "\n  ".join(skipped))
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
        samples_data: list[dict[str, Any]] = []
        sampled_frame_indices: list[int] = []
        instruction = ""

        num_hist = getattr(dataset, 'num_history_sample', 8)

        for frame_idx, dataset_idx in items:
            sample = dataset[dataset_idx]
            instruction = sample['text']
            sampled_frame_indices.append(frame_idx)

            current_t = frame_idx
            available = current_t
            if available <= 0:
                hist_indices = []
            elif num_hist == -1 or available <= num_hist:
                hist_indices = list(range(0, current_t))
            else:
                hist_indices = np.linspace(0, current_t - 1, num_hist, dtype=int).tolist()

            result = infer_sample(
                model,
                sample,
                device=device,
                amp_type=cfg.get('optim', {}).get('amp', 'bf16'),
            )
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

            if args.vis_threshold is not None:
                for v in range(4):
                    if vis_scores[v] < args.vis_threshold:
                        gated_agg[v] = 0.0

            if args.peak_ratio is not None:
                for v in range(4):
                    vmean = float(gated_agg[v].mean())
                    vmax = float(gated_agg[v].max())
                    if vmean < 1e-10 or vmax / vmean < args.peak_ratio:
                        gated_agg[v] = 0.0

            if gt_vis_raw is not None:
                gt_vis_4 = gt_vis_raw.float().max(dim=0).values.numpy() if gt_vis_raw.dim() == 2 else gt_vis_raw.float().numpy()
            else:
                gt_vis_4 = (gt_agg.max(axis=(-2, -1)) > 0).astype(float)

            views_np = sample['current_views'].cpu().numpy().transpose(0, 2, 3, 1)

            logger.info(
                "  frame=%4d N=%d  hist=%s  gt_max=[%.2f,%.2f,%.2f,%.2f]  "
                "gated_max=[%.4f,%.4f,%.4f,%.4f]  vis=[%.2f,%.2f,%.2f,%.2f]",
                frame_idx, n_hist, hist_indices,
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
                'history_indices': hist_indices,
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

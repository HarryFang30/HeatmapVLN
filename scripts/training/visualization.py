"""
Heatmap prediction visualization utilities.
"""

import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def _should_use_gpu_gt(batch: dict, gpu_heatmap_computer) -> bool:
    if gpu_heatmap_computer is None or 'history_poses' not in batch:
        return False
    return 'current_views' not in batch


def _select_primary_heatmap_slice(heatmaps: torch.Tensor) -> torch.Tensor:
    if heatmaps.dim() == 5:
        return heatmaps[:, 0, 0]
    if heatmaps.dim() == 4 and heatmaps.shape[1] == 4:
        return heatmaps[:, 0]
    if heatmaps.dim() == 4:
        return heatmaps[:, -1]
    return heatmaps


def visualize_heatmap_predictions(
    model: nn.Module,
    batch: dict,
    output: dict,
    epoch: int,
    step: int,
    output_dir: Path,
    num_samples: int = 2,
    gt_heatmap_override: torch.Tensor | None = None,
):
    """Visualize heatmap predictions.

    Args:
        gt_heatmap_override: When defer_heatmap_to_gpu is used, pass the GPU-computed
            GT heatmap to replace batch['heatmap'] (which is a zero placeholder).
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    VIEW_LABELS = ["Front", "Right", "Back", "Left"]

    try:
        gt_heatmaps = gt_heatmap_override if gt_heatmap_override is not None else batch['heatmap']
        pred_heatmaps = output.get('heatmaps')

        if pred_heatmaps is None:
            return

        has_panoramic = 'current_views' in batch
        B = min(num_samples, batch['current_frame'].shape[0])

        if has_panoramic:
            pred_vis_raw = output.get('visibility')
            gated_heatmaps = output.get('heatmaps_gated')
            batch_gt_vis = batch.get('gt_visibility')

            total_rows = B * 4
            fig, axes = plt.subplots(total_rows, 4, figsize=(16, 4 * total_rows))
            if total_rows == 1:
                axes = axes[np.newaxis, :]

            for b in range(B):
                views = batch['current_views'][b]
                row_offset = b * 4

                gt_b = gt_heatmaps[b]
                if gt_b.dim() == 4:
                    gt_4 = gt_b.max(dim=0).values
                elif gt_b.dim() == 3 and gt_b.shape[0] == 4:
                    gt_4 = gt_b
                else:
                    gt_4 = gt_b.unsqueeze(0).expand(4, -1, -1)

                gated_4 = None
                if gated_heatmaps is not None:
                    if gated_heatmaps.dim() == 5:
                        gated_4 = gated_heatmaps[b].max(dim=0).values
                    elif gated_heatmaps.dim() == 4:
                        gated_4 = gated_heatmaps[b]

                if gated_4 is None:
                    if pred_heatmaps.dim() == 5:
                        pred_b = pred_heatmaps[b]
                    elif pred_heatmaps.dim() == 4 and pred_heatmaps.shape[1] == 4:
                        pred_b = pred_heatmaps[b].unsqueeze(0)
                    else:
                        pred_b = pred_heatmaps[b].unsqueeze(0).unsqueeze(0).expand(1, 4, -1, -1)
                    N_h, _, Hm, Wm = pred_b.shape
                    sig = pred_b.detach().float().clamp(1e-6, 1 - 1e-6)
                    logits = torch.logit(sig)
                    probs = torch.softmax(logits.reshape(N_h, 4, -1), dim=-1).reshape(N_h, 4, Hm, Wm)
                    if pred_vis_raw is not None:
                        if pred_vis_raw.dim() == 3:
                            vis_gate = torch.sigmoid(pred_vis_raw[b].detach().float())
                        else:
                            vis_gate = torch.sigmoid(pred_vis_raw[b].detach().float()).unsqueeze(0)
                        probs = probs * vis_gate[:, :, None, None]
                    gated_4 = probs.max(dim=0).values

                if pred_vis_raw is not None:
                    if pred_vis_raw.dim() == 3:
                        vis_scores = torch.sigmoid(pred_vis_raw[b].detach().float()).max(dim=0).values.cpu().numpy()
                    else:
                        vis_scores = torch.sigmoid(pred_vis_raw[b].detach().float()).cpu().numpy()
                else:
                    vis_scores = np.ones(4)

                if batch_gt_vis is not None:
                    if batch_gt_vis.dim() == 3:
                        gt_vis_4 = batch_gt_vis[b].float().max(dim=0).values.cpu().numpy()
                    else:
                        gt_vis_4 = batch_gt_vis[b].float().cpu().numpy()
                else:
                    gt_vis_4 = (gt_4.float().amax(dim=(-2, -1)).cpu().numpy() > 0).astype(float)

                N_hist_count = gt_b.shape[0] if gt_b.dim() == 4 else 1

                gated_shared_vmax = max(float(gated_4.max()), 1e-8)

                for v in range(4):
                    r = row_offset + v
                    rgb = views[v].cpu().numpy().transpose(1, 2, 0)
                    rgb = np.clip(rgb, 0, 1)
                    axes[r, 0].imshow(rgb)
                    label = f"S{b} {VIEW_LABELS[v]}" if v == 0 else VIEW_LABELS[v]
                    axes[r, 0].set_title(label, fontweight='bold')
                    axes[r, 0].axis('off')

                    gt_hm = gt_4[v].float().cpu().numpy()
                    axes[r, 1].imshow(gt_hm, cmap='inferno', vmin=0, vmax=max(gt_hm.max(), 0.01))
                    axes[r, 1].set_title(f"GT (max={gt_hm.max():.2f})")
                    axes[r, 1].axis('off')

                    gated_v = gated_4[v].detach().float().cpu().numpy()
                    axes[r, 2].imshow(gated_v, cmap='inferno', vmin=0, vmax=gated_shared_vmax)
                    peak_ratio = float(gated_v.max()) / (1.0 / gated_v.size) if gated_v.size > 0 else 0
                    axes[r, 2].set_title(f"Gated (vis={vis_scores[v]:.2f}, {peak_ratio:.1f}×uni)")
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
                            f"Sample {b}\n(N={N_hist_count})",
                            fontsize=12, fontweight='bold', rotation=0,
                            labelpad=60, va='center',
                        )

            plt.suptitle(f"Epoch {epoch}, Step {step} — {B} samples, max-agg", fontsize=14)
            plt.tight_layout(rect=[0, 0, 1, 0.98])
            save_path = output_dir / f"e{epoch:03d}_s{step:05d}.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            return save_path

        else:
            current_frames = batch['current_frame']
            pred_heatmaps_2d = _select_primary_heatmap_slice(pred_heatmaps)
            gt_heatmaps_2d = _select_primary_heatmap_slice(gt_heatmaps)

            fig, axes = plt.subplots(B, 3, figsize=(12, 4 * B))
            if B == 1:
                axes = axes.reshape(1, -1)
            for i in range(B):
                rgb = current_frames[i].cpu().numpy().transpose(1, 2, 0)
                rgb = np.clip(rgb, 0, 1)
                axes[i, 0].imshow(rgb)
                axes[i, 0].set_title("Input Frame")
                axes[i, 0].axis('off')
                gt_hm = gt_heatmaps_2d[i].cpu().numpy()
                axes[i, 1].imshow(gt_hm, cmap='inferno', vmin=0, vmax=1)
                axes[i, 1].set_title(f"GT (max={gt_hm.max():.2f})")
                axes[i, 1].axis('off')
                pred_sig = pred_heatmaps_2d[i].detach().float()
                _lg = torch.logit(pred_sig.clamp(1e-6, 1 - 1e-6))
                pred_prob = torch.softmax(_lg.reshape(-1), dim=0).reshape_as(pred_sig).cpu().numpy()
                pred_vmax = max(pred_prob.max(), 1e-6)
                axes[i, 2].imshow(pred_prob, cmap='inferno', vmin=0, vmax=pred_vmax)
                pr = pred_prob.max() / (1.0 / (pred_prob.shape[0] * pred_prob.shape[1]))
                axes[i, 2].set_title(f"Pred ({pr:.0f}× unif)")
                axes[i, 2].axis('off')

            plt.suptitle(f"Epoch {epoch}, Step {step}")
            plt.tight_layout()
            save_path = output_dir / f"e{epoch:03d}_s{step:05d}.png"
            plt.savefig(save_path, dpi=100, bbox_inches='tight')
            plt.close(fig)
            return save_path

    except Exception as e:
        logger.warning(f"Visualization failed: {e}")
        return None

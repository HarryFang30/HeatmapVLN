"""
Training curve plotter — generates loss/lr PNG charts and JSON history.
"""

import json
from pathlib import Path


class TrainingPlotter:
    """Training curve plotter."""

    def __init__(self, out_dir: Path, figsize: tuple[int, int] = (14, 10)):
        self.out_dir = Path(out_dir)
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.figsize = figsize

        self.history = {
            'epoch': [],
            'stage': [],
            'train_loss': [],
            'val_loss': [],
            'train_heatmap_loss': [],
            'val_heatmap_loss': [],
            'lr': [],
            'is_best': [],
        }

        self.stage_boundaries = []
        self.current_stage = None

    def update(
        self,
        epoch: int,
        stage_name: str,
        train_metrics: dict[str, float],
        val_metrics: dict[str, float],
        lr: float | None = None,
        is_best: bool = False,
    ):
        if stage_name != self.current_stage:
            if self.current_stage is not None:
                self.stage_boundaries.append(len(self.history['epoch']))
            self.current_stage = stage_name

        self.history['epoch'].append(epoch)
        self.history['stage'].append(stage_name)
        self.history['train_loss'].append(train_metrics.get('total_loss', 0))
        self.history['val_loss'].append(val_metrics.get('val_loss', 0))
        self.history['train_heatmap_loss'].append(train_metrics.get('heatmap_loss', 0))
        self.history['val_heatmap_loss'].append(val_metrics.get('val_heatmap_loss', 0))
        self.history['lr'].append(lr or 0)
        self.history['is_best'].append(is_best)

        self.save_plot()

    def save_plot(self):
        if len(self.history['epoch']) == 0:
            return

        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        epochs = self.history['epoch']

        fig, axes = plt.subplots(2, 2, figsize=self.figsize)
        fig.suptitle('VLN Training Progress', fontsize=14, fontweight='bold')

        def draw_stage_lines(ax):
            for idx in self.stage_boundaries:
                if idx < len(epochs):
                    ax.axvline(x=epochs[idx], color='gray', linestyle='--', alpha=0.5, linewidth=1)

        # Total Loss
        ax1 = axes[0, 0]
        ax1.plot(epochs, self.history['train_loss'], 'b-', label='Train Loss', linewidth=1.5)
        ax1.plot(epochs, self.history['val_loss'], 'r-', label='Val Loss', linewidth=1.5)

        best_indices = [i for i, is_best in enumerate(self.history['is_best']) if is_best]
        if best_indices:
            best_epochs = [epochs[i] for i in best_indices]
            best_vals = [self.history['val_loss'][i] for i in best_indices]
            ax1.scatter(best_epochs, best_vals, c='gold', marker='*', s=100, zorder=5, label='Best Model')

        draw_stage_lines(ax1)
        ax1.set_xlabel('Epoch')
        ax1.set_ylabel('Loss')
        ax1.set_title('Total Loss')
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)

        # Heatmap Loss
        ax2 = axes[0, 1]
        ax2.plot(epochs, self.history['train_heatmap_loss'], 'b-', label='Train Heatmap', linewidth=1.5)
        ax2.plot(epochs, self.history['val_heatmap_loss'], 'r-', label='Val Heatmap', linewidth=1.5)
        draw_stage_lines(ax2)
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Loss')
        ax2.set_title('Heatmap Loss')
        ax2.legend(loc='upper right')
        ax2.grid(True, alpha=0.3)

        # Reserved placeholder
        ax3 = axes[1, 0]
        ax3.set_visible(False)

        # Learning Rate
        ax4 = axes[1, 1]
        if any(lr > 0 for lr in self.history['lr']):
            ax4.plot(epochs, self.history['lr'], 'g-', linewidth=1.5)
            ax4.set_yscale('log')
        draw_stage_lines(ax4)
        ax4.set_xlabel('Epoch')
        ax4.set_ylabel('Learning Rate')
        ax4.set_title('Learning Rate Schedule')
        ax4.grid(True, alpha=0.3)

        if self.stage_boundaries:
            unique_stages = []
            seen = set()
            for s in self.history['stage']:
                if s not in seen:
                    unique_stages.append(s)
                    seen.add(s)
            stage_text = " → ".join(unique_stages)
            fig.text(0.5, 0.02, f"Stages: {stage_text}", ha='center', fontsize=10, style='italic')

        plt.tight_layout(rect=[0, 0.03, 1, 0.97])

        save_path = self.out_dir / 'curves.png'
        plt.savefig(save_path, dpi=120, bbox_inches='tight')
        plt.close(fig)

        json_path = self.out_dir / 'history.json'
        with open(json_path, 'w') as f:
            json.dump(self.history, f, indent=2)

    def get_summary(self) -> dict:
        if not self.history['epoch']:
            return {}

        best_idx = None
        best_val = float('inf')
        for i, (val, is_best) in enumerate(zip(self.history['val_loss'], self.history['is_best'])):
            if is_best and val < best_val:
                best_val = val
                best_idx = i

        return {
            'total_epochs': len(self.history['epoch']),
            'best_epoch': self.history['epoch'][best_idx] if best_idx is not None else None,
            'best_val_loss': best_val if best_idx is not None else None,
            'final_train_loss': self.history['train_loss'][-1],
            'final_val_loss': self.history['val_loss'][-1],
        }

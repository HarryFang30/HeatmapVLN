"""
Training time tracker and ETA estimator.
"""

import time
from datetime import timedelta

import numpy as np


class TrainingTimer:
    """Training time and ETA estimator."""

    def __init__(self, total_epochs: int):
        self.total_epochs = total_epochs
        self.start_time = None
        self.epoch_times = []
        self.epoch_start_time = None

    def start(self):
        self.start_time = time.time()
        self.epoch_start_time = time.time()

    def start_epoch(self):
        self.epoch_start_time = time.time()

    def end_epoch(self):
        if self.epoch_start_time is None:
            return
        elapsed = time.time() - self.epoch_start_time
        self.epoch_times.append(elapsed)
        self.epoch_start_time = time.time()

    def get_eta(self, current_epoch: int, total_epochs: int) -> str:
        if not self.epoch_times:
            return "计算中..."

        avg_epoch_time = np.mean(self.epoch_times[-5:])
        remaining_epochs = total_epochs - current_epoch
        eta_seconds = avg_epoch_time * remaining_epochs

        if eta_seconds < 60:
            return f"{eta_seconds:.0f}秒"
        elif eta_seconds < 3600:
            return f"{eta_seconds/60:.1f}分钟"
        else:
            return f"{eta_seconds/3600:.1f}小时"

    def get_epoch_time(self) -> str:
        if not self.epoch_times:
            return "N/A"
        last_time = self.epoch_times[-1]
        if last_time < 60:
            return f"{last_time:.1f}s"
        else:
            return f"{last_time/60:.1f}min"

    def get_total_elapsed(self) -> str:
        if self.start_time is None:
            return "N/A"
        elapsed = time.time() - self.start_time
        return str(timedelta(seconds=int(elapsed)))

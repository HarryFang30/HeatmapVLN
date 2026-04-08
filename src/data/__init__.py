"""
Data loading and preprocessing utilities for VLN datasets
==========================================================

Active components:
- vln_sliding_window_dataset: Core training dataset (滑动窗口)
- packing_collator: Sequence Packing Collator（保留的旧实现，默认不用）
"""

# Training dataset classes
from .vln_sliding_window_dataset import (
    VLNSlidingWindowDataset,
    create_sliding_window_dataloader
)

# Sequence Packing
from .packing_collator import PackingCollatorForVLN

__all__ = [
    # Training Datasets
    'VLNSlidingWindowDataset',
    'create_sliding_window_dataloader',
    # Sequence Packing
    'PackingCollatorForVLN',
]

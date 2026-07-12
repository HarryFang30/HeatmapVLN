"""Helpers for identifying the real (unpadded) history axis of a sample.

Some panoramic stages deliberately set ``load_history_frames=false`` and keep
one dummy front-view frame to satisfy the legacy batch interface.  That dummy
axis is not the navigation-history axis: the panoramic images, per-history
heatmaps, and relative poses still contain the real number of history steps.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any


def infer_history_length(sample: Mapping[str, Any]) -> int:
    """Return the real per-history length for one uncollated sample.

    Prefer tensors that are consumed by the panoramic heatmap branch.  Fall
    back to ``history_frames`` only for legacy/non-panoramic samples where no
    stronger history-aligned field is available.
    """

    history_panoramas = sample.get("history_panoramas")
    if history_panoramas is not None and getattr(history_panoramas, "ndim", 0) >= 1:
        return int(history_panoramas.shape[0])

    # Panoramic heatmap targets are [K, 4, H, W].  A legacy aggregate
    # heatmap is [H, W] and therefore must not be interpreted as K=H.
    heatmap = sample.get("heatmap")
    if heatmap is not None and getattr(heatmap, "ndim", 0) >= 4:
        return int(heatmap.shape[0])

    gt_visibility = sample.get("gt_visibility")
    if gt_visibility is not None and getattr(gt_visibility, "ndim", 0) >= 2:
        return int(gt_visibility.shape[0])

    history_rel_poses = sample.get("history_rel_poses")
    if history_rel_poses is not None and getattr(history_rel_poses, "ndim", 0) >= 1:
        return int(history_rel_poses.shape[0])

    history_poses = sample.get("history_poses")
    if history_poses is not None and getattr(history_poses, "ndim", 0) >= 1:
        return int(history_poses.shape[0])

    history_frames = sample.get("history_frames")
    if history_frames is None or getattr(history_frames, "ndim", 0) < 1:
        raise ValueError(
            "Cannot infer history length: sample has no history_panoramas, "
            "per-history heatmap/poses, or history_frames"
        )
    return int(history_frames.shape[0])

"""
DataLoader collate function for sliding-window / trajectory datasets.
"""

from typing import Any

import torch


def _heatmap_history_length(sample: dict) -> int:
    """Return the real number of heatmap histories for one sample.

    ``history_frames`` can intentionally be a one-frame dummy tensor when
    panoramic heatmap training sets ``load_single_view_history_frames=false``.
    Panoramic ``history_panoramas`` remain loaded independently.  Building
    the mask from that tensor silently masks or unmasks the wrong histories.
    Prefer occurrence-aligned panoramic/pose targets and fall back to frames.
    """
    for key in ("history_panoramas", "history_rel_poses", "gt_visibility"):
        value = sample.get(key)
        if torch.is_tensor(value) and value.ndim >= 1:
            return int(value.shape[0])
    heatmap = sample.get("heatmap")
    if torch.is_tensor(heatmap) and heatmap.ndim >= 4:
        return int(heatmap.shape[0])
    return int(sample["history_frames"].shape[0])


def _pad_and_stack(batch: list[dict], key: str) -> torch.Tensor:
    """Stack tensors that may differ in the first (history) dimension,
    padding shorter ones with zeros to match the longest."""
    tensors = [s[key] for s in batch]
    if all(t.shape == tensors[0].shape for t in tensors):
        return torch.stack(tensors, dim=0)
    max_n = max(t.shape[0] for t in tensors)
    padded = []
    for t in tensors:
        if t.shape[0] < max_n:
            pad_shape = (max_n - t.shape[0], *t.shape[1:])
            t = torch.cat([t, torch.zeros(pad_shape, dtype=t.dtype)], dim=0)
        padded.append(t)
    return torch.stack(padded, dim=0)


def collate_fn(batch: list[dict]) -> dict[str, Any]:
    max_K = max(s['history_frames'].shape[0] for s in batch)
    heatmap_lengths = [_heatmap_history_length(sample) for sample in batch]
    max_heatmap_K = max(heatmap_lengths)

    history_frames_padded = []

    for s in batch:
        frames = s['history_frames']
        K = frames.shape[0]

        if max_K > K:
            pad_size = max_K - K
            pad_frames = frames[-1:].repeat(pad_size, 1, 1, 1)
            frames_padded = torch.cat([frames, pad_frames], dim=0)
        else:
            frames_padded = frames

        history_frames_padded.append(frames_padded)

    history_frames = torch.stack(history_frames_padded, dim=0)
    history_mask = torch.stack([
        torch.cat([
            torch.ones(length),
            torch.zeros(max_heatmap_K - length),
        ])
        for length in heatmap_lengths
    ], dim=0)
    current_frame = torch.stack([s['current_frame'] for s in batch], dim=0)

    heatmap = _pad_and_stack(batch, 'heatmap')

    action = torch.stack([s['action'] for s in batch], dim=0)
    action_valid = torch.tensor([s['action_valid'] for s in batch])
    discrete_action = torch.tensor([s.get('discrete_action', 1) for s in batch])
    is_stop = torch.tensor([s.get('is_stop', 0.0) for s in batch])
    text = [s['text'] for s in batch]

    result = {
        'history_frames': history_frames,
        'history_mask': history_mask,
        'current_frame': current_frame,
        'heatmap': heatmap,
        'action': action,
        'action_valid': action_valid,
        'discrete_action': discrete_action,
        'is_stop': is_stop,
        'text': text,
    }

    if 'current_views' in batch[0]:
        result['current_views'] = torch.stack([s['current_views'] for s in batch], dim=0)
    if 'history_panoramas' in batch[0]:
        result['history_panoramas'] = _pad_and_stack(batch, 'history_panoramas')
    if 'gt_visibility' in batch[0]:
        result['gt_visibility'] = _pad_and_stack(batch, 'gt_visibility')

    if 'is_flipped' in batch[0]:
        result['is_flipped'] = torch.tensor([s.get('is_flipped', False) for s in batch], dtype=torch.bool)

    if 'trajectory' in batch[0]:
        result['trajectory'] = torch.stack([s['trajectory'] for s in batch], dim=0)
        trajectory_valid = [s.get('trajectory_valid', 0.0) for s in batch]
        if torch.is_tensor(trajectory_valid[0]):
            result['trajectory_valid'] = torch.stack(trajectory_valid, dim=0)
        else:
            result['trajectory_valid'] = torch.tensor(trajectory_valid)
        result['progress'] = torch.tensor([s.get('progress', 0.0) for s in batch])

    if 'history_rel_poses' in batch[0]:
        max_K_rel = max(s['history_rel_poses'].shape[0] for s in batch)
        rel_poses_padded = []
        for s in batch:
            rp = s['history_rel_poses']
            if rp.shape[0] < max_K_rel:
                pad = torch.zeros(max_K_rel - rp.shape[0], rp.shape[1], dtype=rp.dtype)
                rp = torch.cat([rp, pad], dim=0)
            rel_poses_padded.append(rp)
        result['history_rel_poses'] = torch.stack(rel_poses_padded, dim=0)

    if 'history_poses' in batch[0]:
        result['history_poses'] = _pad_and_stack(batch, 'history_poses')
        result['current_pose'] = torch.stack([s['current_pose'] for s in batch], dim=0)
        if 'current_depth' in batch[0]:
            result['current_depth'] = torch.stack([s['current_depth'] for s in batch], dim=0)
        if 'intrinsics' in batch[0]:
            result['intrinsics'] = torch.stack([s['intrinsics'] for s in batch], dim=0)

    return result

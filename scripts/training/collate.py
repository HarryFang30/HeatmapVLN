"""
DataLoader collate function for sliding-window / trajectory datasets.
"""

from typing import Any

import torch


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

    history_frames_padded = []
    history_mask = []

    for s in batch:
        frames = s['history_frames']
        K = frames.shape[0]

        if max_K > K:
            pad_size = max_K - K
            pad_frames = frames[-1:].repeat(pad_size, 1, 1, 1)
            frames_padded = torch.cat([frames, pad_frames], dim=0)
            mask = torch.cat([torch.ones(K), torch.zeros(pad_size)])
        else:
            frames_padded = frames
            mask = torch.ones(K)

        history_frames_padded.append(frames_padded)
        history_mask.append(mask)

    history_frames = torch.stack(history_frames_padded, dim=0)
    history_mask = torch.stack(history_mask, dim=0)
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
        result['trajectory_valid'] = torch.tensor([s.get('trajectory_valid', 0.0) for s in batch])
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

    return result

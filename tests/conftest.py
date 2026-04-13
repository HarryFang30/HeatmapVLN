"""Shared test fixtures — minimal configs and dummy samples.

All fixtures are pure in-memory and require no GPU, data files, or
model weights.
"""

import sys
from pathlib import Path

import pytest
import torch

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


@pytest.fixture()
def minimal_cfg():
    """Return the smallest valid config dict that passes Pydantic validation."""
    return {
        "seed": 42,
        "data": {
            "root": "/tmp/fake_data",
            "image_size": [256, 256],
            "init_hm_size": [64, 64],
            "dataset_type": "sliding_window",
            "sliding_window": {
                "min_history": 5,
                "num_history_sample": 8,
            },
            "trajectory": {
                "min_history": 5,
                "num_history_sample": 8,
            },
        },
        "model": {
            "type": "vln_pipeline",
            "device": "cpu",
        },
        "optim": {
            "batch_size": 2,
            "learning_rate": 1e-4,
            "warmup_ratio": 0.05,
        },
        "loss": {
            "heatmap_loss_type": "heatmap_vln",
        },
        "training": {
            "stages": [
                {
                    "name": "test_stage",
                    "epochs": 1,
                    "hm_size": [64, 64],
                }
            ]
        },
        "gpu": {"devices": [0]},
        "log": {
            "out_dir": "/tmp/test_output",
        },
        "validation": {},
    }


def _make_dummy_sample(
    H: int = 256,
    W: int = 256,
    K: int = 8,
    hm_h: int = 64,
    hm_w: int = 64,
    with_trajectory: bool = False,
):
    """Build a single sample dict mimicking dataset __getitem__ output."""
    sample = {
        "history_frames": torch.randn(K, 3, H, W),
        "current_frame": torch.randn(3, H, W),
        "heatmap": torch.randn(hm_h, hm_w),
        "action": torch.randn(2),
        "action_valid": 1.0,
        "discrete_action": 1,
        "is_stop": 0.0,
        "text": "Turn right and walk forward.",
        "history_rel_poses": torch.randn(K, 4),
    }
    if with_trajectory:
        sample["trajectory"] = torch.randn(24, 3)
        sample["trajectory_valid"] = 1.0
        sample["progress"] = 0.5
    return sample


@pytest.fixture()
def dummy_sample():
    """A single dummy dataset sample (no trajectory)."""
    return _make_dummy_sample()


@pytest.fixture()
def dummy_sample_traj():
    """A single dummy dataset sample with trajectory fields."""
    return _make_dummy_sample(with_trajectory=True)

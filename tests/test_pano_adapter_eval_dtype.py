from __future__ import annotations

import argparse
from pathlib import Path

import torch
from scripts.evaluation.eval_pano_latent_adapter import _load_adapter_from_checkpoint

from src.models.adapters import PanoLatentSpaceAdapter


def test_checkpoint_loader_uses_requested_model_dtype(tmp_path: Path):
    source = PanoLatentSpaceAdapter(dim=8, hidden_dim=4, dropout=0.0)
    checkpoint = tmp_path / "adapter.pth"
    torch.save(
        {
            "adapter_type": "pano_latent_space",
            "adapter_state_dict": source.state_dict(),
            "args": {"adapter_dropout": 0.0},
        },
        checkpoint,
    )

    loaded, _ = _load_adapter_from_checkpoint(
        checkpoint,
        dim=8,
        fallback_args=argparse.Namespace(),
        device=torch.device("cpu"),
        dtype=torch.bfloat16,
    )

    assert {parameter.dtype for parameter in loaded.parameters()} == {torch.bfloat16}
    assert loaded.training is False


def test_checkpoint_loader_keeps_fp32_when_dtype_is_unspecified(tmp_path: Path):
    source = PanoLatentSpaceAdapter(dim=8, hidden_dim=4, dropout=0.0)
    checkpoint = tmp_path / "adapter.pth"
    torch.save(
        {
            "adapter_type": "pano_latent_space",
            "adapter_state_dict": source.state_dict(),
            "args": {"adapter_dropout": 0.0},
        },
        checkpoint,
    )

    loaded, _ = _load_adapter_from_checkpoint(
        checkpoint,
        dim=8,
        fallback_args=argparse.Namespace(),
        device=torch.device("cpu"),
    )

    assert {parameter.dtype for parameter in loaded.parameters()} == {torch.float32}

"""Contract tests for the ``geometry`` memory-token arm (EXP-17).

* every mode owns the same parameter tensors, so the arms share one budget;
* pose tokens are the Past Head's own sinusoidal encoding of (forward, left,
  cos yaw, sin yaw), projected, with padded slots reading the ``absent`` vector;
* ``force_no_pose`` blanks every slot and ``pose_dropout`` blanks whole samples
  during training only, so "read without odometry" is in-distribution.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
import torch

_MODULE = Path(__file__).resolve().parents[1] / "src/models/system2_memory.py"


def _load():
    spec = importlib.util.spec_from_file_location("_system2_memory_geometry", _MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


system2_memory = _load()
System2MemoryTokens = system2_memory.System2MemoryTokens


def _module(mode: str, **kwargs) -> System2MemoryTokens:
    torch.manual_seed(0)
    return System2MemoryTokens(memory_dim=256, embed_dim=32, num_tokens=4, mode=mode, **kwargs)


def test_every_mode_owns_the_same_parameter_tensors() -> None:
    names = {mode: sorted(k for k, _ in _module(mode).named_parameters()) for mode in ("memory", "constant", "geometry")}
    assert names["memory"] == names["constant"] == names["geometry"]
    assert "geometry_projection.weight" in names["geometry"]
    assert system2_memory.pose_pe_dim(16) == 132


def test_geometry_tokens_have_the_prompt_shape_and_blank_padded_slots() -> None:
    module = _module("geometry").eval()
    poses = torch.zeros(2, 4, 4)
    poses[:, :, 2] = 1.0  # cos yaw = 1
    poses[0, 1] = torch.tensor([3.0, -1.0, 0.0, 1.0])
    mask = torch.tensor([[1, 1, 0, 0], [1, 0, 0, 0]], dtype=torch.bool)
    tokens = module(None, mask, history_rel_poses=poses)
    assert tokens.shape == (2, 4, 32)
    absent = module.absent_embedding.detach()
    slot = module.slot_embedding.detach()
    assert torch.allclose(tokens[0, 2], absent + slot[2])
    assert torch.allclose(tokens[1, 3], absent + slot[3])
    assert not torch.allclose(tokens[0, 1], absent + slot[1])


def test_force_no_pose_blanks_every_slot() -> None:
    module = _module("geometry").eval()
    poses = torch.randn(3, 4, 4)
    mask = torch.ones(3, 4, dtype=torch.bool)
    tokens = module(None, mask, history_rel_poses=poses, force_no_pose=True)
    expected = module.absent_embedding.detach().view(1, 1, -1) + module.slot_embedding.detach().unsqueeze(0)
    assert torch.allclose(tokens, expected.expand(3, -1, -1))


def test_pose_dropout_blanks_whole_samples_only_while_training() -> None:
    module = _module("geometry", pose_dropout=0.5)
    poses = torch.randn(64, 4, 4)
    mask = torch.ones(64, 4, dtype=torch.bool)
    blank = (module.absent_embedding.detach().view(1, 1, -1) + module.slot_embedding.detach().unsqueeze(0)).expand(64, -1, -1)
    torch.manual_seed(1)
    module.train()
    tokens = module(None, mask, history_rel_poses=poses)
    row_blank = torch.isclose(tokens, blank).all(dim=-1).all(dim=-1)
    assert 8 < int(row_blank.sum()) < 56  # some rows blanked, some kept
    partial = torch.isclose(tokens, blank).all(dim=-1).any(dim=-1) & ~row_blank
    assert not bool(partial.any())  # never a partially blanked row
    module.eval()
    tokens_eval = module(None, mask, history_rel_poses=poses)
    assert not bool(torch.isclose(tokens_eval, blank).all(dim=-1).all(dim=-1).any())


def test_geometry_mode_rejects_wrong_inputs_and_other_modes_ignore_poses() -> None:
    module = _module("geometry").eval()
    with pytest.raises(ValueError, match="history_rel_poses"):
        module(None, torch.ones(1, 4, dtype=torch.bool))
    with pytest.raises(ValueError, match="slots"):
        module(None, torch.ones(1, 3, dtype=torch.bool), history_rel_poses=torch.zeros(1, 3, 4))
    with pytest.raises(ValueError, match="non-finite"):
        module(None, torch.ones(1, 4, dtype=torch.bool), history_rel_poses=torch.full((1, 4, 4), float("nan")))
    constant = _module("constant").eval()
    tokens = constant(None, None, history_rel_poses=torch.randn(2, 4, 4))
    assert tokens.shape == (2, 4, 32)
    assert torch.allclose(tokens[0], tokens[1])
    with pytest.raises(ValueError, match="pose_dropout"):
        _module("geometry", pose_dropout=1.0)


def test_sinusoidal_encoding_matches_the_past_heads_layout() -> None:
    poses = torch.tensor([[[1.0, 2.0, 0.6, 0.8]]])
    encoded = system2_memory.sinusoidal_pose_encoding(poses, num_freqs=2, max_spatial_range=10.0)
    assert encoded.shape == (1, 1, 4 * (1 + 2 * 2))
    assert torch.allclose(encoded[0, 0, :4], poses[0, 0] / 10.0)

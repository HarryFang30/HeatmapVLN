"""Contract tests for the EXP-15 pose-noise sweep.

The sweep's job is to tell a real collapse from an artefact of the perturbation
itself, so the noise model is what gets pinned here:

- yaw must be perturbed by *rotating* ``(cos, sin)``, keeping the unit norm
  exact.  Adding Gaussian noise to each component independently would let a
  readout detect the corruption instead of the pose error it stands for, and
  the sweep would report a collapse that means nothing.
- zero noise must be a byte-exact copy, or the level-0 row would not be
  comparable with EXP-13-A's table.
- drift must scale with age, because odometry error accumulates.
- arms that do not depend on the poses must be refused, since sweeping them
  measures refit noise and nothing else.
"""

from __future__ import annotations

import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

_REPO_ROOT = Path(__file__).resolve().parents[1]


def _load_tool() -> types.ModuleType:
    path = _REPO_ROOT / "scripts/tools/probe_pose_noise_readout.py"
    spec = importlib.util.spec_from_file_location("_exp15_pose_noise_readout", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


tool = _load_tool()


def _poses(n: int = 64, k: int = 8) -> np.ndarray:
    rng = np.random.default_rng(0)
    yaw = rng.uniform(-np.pi, np.pi, size=(n, k))
    return np.stack(
        [
            rng.normal(0, 2.0, size=(n, k)),
            rng.normal(0, 2.0, size=(n, k)),
            np.cos(yaw),
            np.sin(yaw),
        ],
        axis=-1,
    ).astype(np.float32)


def test_zero_noise_is_an_exact_copy() -> None:
    poses = _poses()
    out = tool.perturb_rel_poses(
        poses,
        translation_m=0.0,
        rotation_deg=0.0,
        ages=np.zeros(poses.shape[:2]),
        drift=False,
        rng=np.random.default_rng(1),
    )
    assert np.array_equal(out, poses)
    assert out is not poses


def test_yaw_stays_on_the_unit_circle_under_rotation_noise() -> None:
    poses = _poses()
    out = tool.perturb_rel_poses(
        poses,
        translation_m=0.0,
        rotation_deg=45.0,
        ages=np.zeros(poses.shape[:2]),
        drift=False,
        rng=np.random.default_rng(2),
    )
    norms = np.linalg.norm(out[:, :, 2:], axis=-1)
    assert np.allclose(norms, 1.0, atol=1e-5)
    # And it actually moved.
    assert not np.allclose(out[:, :, 2:], poses[:, :, 2:])
    # Translation was untouched.
    assert np.array_equal(out[:, :, :2], poses[:, :, :2])


def test_translation_noise_has_the_requested_scale_and_leaves_yaw_alone() -> None:
    poses = _poses(n=4000)
    sigma = 0.3
    out = tool.perturb_rel_poses(
        poses,
        translation_m=sigma,
        rotation_deg=0.0,
        ages=np.zeros(poses.shape[:2]),
        drift=False,
        rng=np.random.default_rng(3),
    )
    delta = out[:, :, :2] - poses[:, :, :2]
    assert delta.std() == pytest.approx(sigma, rel=0.05)
    assert np.array_equal(out[:, :, 2:], poses[:, :, 2:])


def test_drift_makes_older_slots_noisier() -> None:
    poses = _poses(n=4000, k=2)
    ages = np.zeros((4000, 2), dtype=np.float32)
    ages[:, 1] = 15.0  # sqrt(1+15) = 4x the sigma of a fresh slot
    out = tool.perturb_rel_poses(
        poses,
        translation_m=0.1,
        rotation_deg=0.0,
        ages=ages,
        drift=True,
        rng=np.random.default_rng(4),
    )
    delta = out[:, :, :2] - poses[:, :, :2]
    fresh, old = delta[:, 0].std(), delta[:, 1].std()
    assert old / fresh == pytest.approx(4.0, rel=0.1)


def test_drift_needs_ages() -> None:
    with pytest.raises(ValueError, match="history_age_steps"):
        tool.perturb_rel_poses(
            _poses(), translation_m=0.1, rotation_deg=0.0, ages=None, drift=True,
            rng=np.random.default_rng(5),
        )


def test_a_malformed_pose_block_is_refused() -> None:
    with pytest.raises(ValueError, match=r"\[N,K,4\]"):
        tool.perturb_rel_poses(
            np.zeros((4, 3), dtype=np.float32), translation_m=0.1, rotation_deg=1.0,
            ages=None, drift=False, rng=np.random.default_rng(6),
        )


def test_only_pose_dependent_arms_may_be_swept() -> None:
    # memory and system2 are replayed from cache; sweeping them would report
    # refit noise as if it were a pose effect.
    assert set(tool.POSE_DEPENDENT) == {"geometry", "system2_geometry"}
    assert "memory" not in tool.POSE_DEPENDENT
    assert "system2" not in tool.POSE_DEPENDENT


def test_the_recipe_is_imported_from_exp13a_not_reimplemented() -> None:
    readout = tool._load_readout_tool()
    for name in ("scene_split", "build_arms", "fit_linear", "score", "predict"):
        assert hasattr(readout, name), name

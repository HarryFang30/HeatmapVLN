"""Contract tests for the EXP-10 H2 token-binding probe.

The probe's verdict is read off how much the future metrics drop when the plan
tokens are permuted, so the dangerous failure is silent: a permutation helper
that shuffles the wrong axis, or is accidentally a no-op, makes every arm equal
``identity`` and produces a confident "H2 denied" for entirely the wrong reason.
These tests pin the permutation semantics directly.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest

torch = pytest.importorskip("torch")

PROBE_PATH = (
    Path(__file__).resolve().parents[1]
    / "scripts"
    / "tools"
    / "probe_future_plan_token_binding.py"
)


def _load_probe():
    """Import the probe without executing its heavy training-stack imports."""
    spec = importlib.util.spec_from_file_location("_h2_probe", PROBE_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    try:
        spec.loader.exec_module(module)
    except ImportError as exc:  # pragma: no cover - depends on the environment
        pytest.skip(f"probe dependencies unavailable: {exc}")
    return module


probe = _load_probe()


def test_fixed_arms_are_derangements_except_identity():
    """Every non-identity fixed arm must move all four vectors."""
    assert probe.FIXED_ARMS["identity"] == (0, 1, 2, 3)
    for name, order in probe.FIXED_ARMS.items():
        assert sorted(order) == [0, 1, 2, 3], f"{name} is not a permutation"
        if name == "identity":
            continue
        fixed_points = [i for i, slot in enumerate(order) if slot == i]
        assert not fixed_points, f"{name} leaves vectors {fixed_points} in place"


def test_random_derangement_never_has_a_fixed_point():
    generator = torch.Generator().manual_seed(0)
    for _ in range(200):
        order = probe.random_derangement(4, generator)
        assert sorted(order.tolist()) == [0, 1, 2, 3]
        assert not bool((order == torch.arange(4)).any())


def test_permute_acts_on_the_token_axis_not_the_batch():
    """Each row must be permuted within itself, leaving batch order intact."""
    # Row b, token t carries the value b*10 + t, so a mix-up between the two
    # axes is visible in the values themselves.
    plan_z = torch.tensor(
        [[[0.0], [1.0], [2.0], [3.0]], [[10.0], [11.0], [12.0], [13.0]]]
    )
    generator = torch.Generator().manual_seed(0)
    out = probe.permute_plan_z(plan_z, "reverse", generator)
    assert out.shape == plan_z.shape
    assert out[0].squeeze(-1).tolist() == [3.0, 2.0, 1.0, 0.0]
    assert out[1].squeeze(-1).tolist() == [13.0, 12.0, 11.0, 10.0]
    # Each row still holds its own values: no cross-row leakage.
    assert set(out[0].squeeze(-1).tolist()) == {0.0, 1.0, 2.0, 3.0}
    assert set(out[1].squeeze(-1).tolist()) == {10.0, 11.0, 12.0, 13.0}


def test_identity_arm_is_a_true_no_op():
    plan_z = torch.randn(3, 4, 8)
    generator = torch.Generator().manual_seed(0)
    out = probe.permute_plan_z(plan_z, "identity", generator)
    assert torch.equal(out, plan_z)


def test_every_non_identity_arm_actually_changes_the_tensor():
    """The regression that would silently produce a false 'H2 denied'."""
    plan_z = torch.randn(4, 4, 8)
    generator = torch.Generator().manual_seed(0)
    for arm in ("reverse", "roll1", "roll2", "random_derangement"):
        out = probe.permute_plan_z(plan_z, arm, generator)
        assert not torch.equal(out, plan_z), f"{arm} left the tensor untouched"
        # A permutation moves values around but never invents or drops any.
        assert torch.allclose(out.sum(dim=1), plan_z.sum(dim=1))


def test_random_derangement_arm_varies_across_rows():
    """Per-sample derangements, not one order reused for the whole batch."""
    plan_z = torch.arange(64, dtype=torch.float32).reshape(4, 4, 4)
    generator = torch.Generator().manual_seed(7)
    orders = set()
    for _ in range(40):
        out = probe.permute_plan_z(plan_z, "random_derangement", generator)
        for row in range(plan_z.shape[0]):
            first_token = out[row, 0, 0].item() - plan_z[row, 0, 0].item()
            orders.add((row, first_token))
    # If a single order were reused for every row and every call there would be
    # exactly one distinct offset per row.
    assert len({offset for _, offset in orders}) > 1


def test_unknown_arm_is_rejected():
    generator = torch.Generator().manual_seed(0)
    with pytest.raises(ValueError, match="unknown arm"):
        probe.permute_plan_z(torch.randn(1, 4, 2), "shuffle_everything", generator)

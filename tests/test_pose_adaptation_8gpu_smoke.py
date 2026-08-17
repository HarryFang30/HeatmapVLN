"""Pure contract tests for the dedicated eight-rank pose-adaptation smoke."""

from __future__ import annotations

import copy
import inspect

import pytest

from scripts.training.pose_adaptation_smoke import (
    EXPECTED_GRADIENT_FAMILY_TENSORS,
    GRADIENT_FAMILIES,
    gather_and_validate_local_audit,
    validate_rank_audits,
)
from scripts.training.train_loop import train_one_epoch


def _gradient_records() -> dict[str, dict[str, object]]:
    records: dict[str, dict[str, object]] = {}
    for family, prefix in GRADIENT_FAMILIES.items():
        for index in range(EXPECTED_GRADIENT_FAMILY_TENSORS[family]):
            records[f"{prefix}{family}_{index}.weight"] = {
                "seen": True,
                "finite": True,
                "nonzero": index == 0,
                "l2": float(index == 0),
                "max_abs": float(index == 0),
            }
    assert len(records) == 34
    return records


def _valid_rank_audits(world_size: int = 8) -> list[dict[str, object]]:
    return [
        {
            "rank": rank,
            "world_size": world_size,
            "identities": [
                f"scene/clip_{rank:06d}@{2 * rank:06d}",
                f"scene/clip_{rank:06d}@{2 * rank + 1:06d}",
            ],
            "providers": ["amb3r_vo_cache", "amb3r_vo_cache"],
            "optimizer_steps": 1,
            "gradient_records": _gradient_records(),
            "post_parameter_digest": "same-model-digest",
            "ema_digest": "same-ema-digest",
            "ema_step_count": 1,
        }
        for rank in range(world_size)
    ]


def test_validates_complete_eight_rank_contract() -> None:
    report = validate_rank_audits(_valid_rank_audits())

    assert report["status"] == "passed"
    assert report["global_unique_identity_count"] == 16
    assert report["optimizer_steps_by_rank"] == [1] * 8
    assert report["gradient_hook_tensors_by_rank"] == [34] * 8
    assert report["post_parameter_digest_unique_count"] == 1
    assert report["ema_digest_unique_count"] == 1
    assert report["gradient_family_tensor_counts"] == {
        "proj_traj": 2,
        "transformer": 24,
        "visibility": 4,
        "coarse_heatmap": 4,
    }
    assert report["checkpoint_hash_locking"] is False
    for family in GRADIENT_FAMILIES:
        assert report["gradient_families_nonzero_on_ranks"][family] == list(range(8))


def test_validates_complete_four_rank_contract() -> None:
    report = validate_rank_audits(
        _valid_rank_audits(world_size=4),
        expected_world_size=4,
    )

    assert report["world_size"] == 4
    assert report["global_identity_count"] == 8
    assert report["global_unique_identity_count"] == 8
    assert report["optimizer_steps_by_rank"] == [1] * 4
    assert report["gradient_hook_tensors_by_rank"] == [34] * 4
    for family in GRADIENT_FAMILIES:
        assert report["gradient_families_nonzero_on_ranks"][family] == list(range(4))


def test_rejects_duplicate_global_identity() -> None:
    audits = _valid_rank_audits()
    audits[7]["identities"] = list(audits[0]["identities"])

    with pytest.raises(RuntimeError, match="globally unique identities"):
        validate_rank_audits(audits)


@pytest.mark.parametrize("field", ["seen", "finite"])
def test_rejects_missing_or_nonfinite_gradient_hook(field: str) -> None:
    audits = _valid_rank_audits()
    first_name = next(iter(audits[3]["gradient_records"]))
    audits[3]["gradient_records"][first_name][field] = False

    with pytest.raises(RuntimeError, match="missing/non-finite gradients"):
        validate_rank_audits(audits)


def test_rejects_zero_gradient_family() -> None:
    audits = _valid_rank_audits()
    prefix = GRADIENT_FAMILIES["visibility"]
    for name, record in audits[2]["gradient_records"].items():
        if name.startswith(prefix):
            record["nonzero"] = False

    with pytest.raises(RuntimeError, match="family visibility"):
        validate_rank_audits(audits)


def test_rejects_wrong_exact_gradient_family_tensor_count() -> None:
    audits = _valid_rank_audits()
    for audit in audits:
        records = audit["gradient_records"]
        removed = next(
            name
            for name in records
            if name.startswith(GRADIENT_FAMILIES["transformer"])
        )
        replacement = f"{GRADIENT_FAMILIES['proj_traj']}unexpected.weight"
        records[replacement] = records.pop(removed)

    with pytest.raises(RuntimeError, match="family proj_traj has 3/2 tensors"):
        validate_rank_audits(audits)


@pytest.mark.parametrize("digest_field", ["post_parameter_digest", "ema_digest"])
def test_rejects_cross_rank_parameter_or_ema_divergence(digest_field: str) -> None:
    audits = copy.deepcopy(_valid_rank_audits())
    audits[5][digest_field] = "diverged"

    with pytest.raises(RuntimeError, match="diverged"):
        validate_rank_audits(audits)


def test_actual_batch_observer_runs_after_skip_max_and_before_provider_forward() -> None:
    """Freeze the observer's safety-critical placement in the production loop."""

    source = inspect.getsource(train_one_epoch)
    loop = source[source.index("for i, batch in enumerate(pbar):") :]
    skip_position = loop.index("if skip_first_n_batches is not None")
    max_position = loop.index("if max_batches is not None and i >= max_batches")
    observer_position = loop.index("actual_batch_observer(i, batch)")
    provider_position = loop.index("assert_required_history_pose_provider(batch, stage_cfg)")
    forward_position = loop.index("output = model(")

    assert skip_position < max_position < observer_position
    assert observer_position < provider_position < forward_position


def test_collective_propagates_local_audit_error_to_every_rank(monkeypatch) -> None:
    import scripts.training.pose_adaptation_smoke as smoke

    monkeypatch.setattr(smoke.dist, "is_available", lambda: True)
    monkeypatch.setattr(smoke.dist, "is_initialized", lambda: True)
    monkeypatch.setattr(smoke.dist, "get_world_size", lambda: 8)

    def fake_all_gather_object(gathered, local_payload):
        for rank in range(8):
            gathered[rank] = {
                "audit": None,
                "error": "ValueError: rank-local digest failed" if rank == 3 else None,
            }

    monkeypatch.setattr(smoke.dist, "all_gather_object", fake_all_gather_object)

    with pytest.raises(RuntimeError, match="rank-local digest failed"):
        gather_and_validate_local_audit({"rank": 0})

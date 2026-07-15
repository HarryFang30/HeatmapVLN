import copy

import pytest
import torch
from scripts.evaluation.rpc_protocol import (
    HEATMAPVLN_RPC_PROTOCOL_VERSION,
    HEATMAPVLN_RPC_SAMPLING_PROTOCOL,
    build_rpc_progress_sampling_contract,
    build_rpc_sampling_metadata,
    validate_rpc_progress_sampling_contract,
    validate_rpc_sampling_metadata,
)


def _metadata(*, scene="17DRP5sb8fy", episode=7, call=2):
    return build_rpc_sampling_metadata(
        protocol_seed=42,
        scene_id=scene,
        episode_id=episode,
        system2_call_index=call,
    )


def _noise_for(metadata: dict) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(metadata["per_call_seed"])
    return torch.randn(16, generator=generator)


def test_same_call_key_produces_same_seed_and_noise():
    first = _metadata()
    second = _metadata()
    global_state = torch.random.get_rng_state().clone()

    assert first == second
    assert first["sampling_protocol"] == HEATMAPVLN_RPC_SAMPLING_PROTOCOL
    assert first["per_call_seed"] == 1126458639812566750
    assert first["seed_sha256"] == ("0fa1fc2a28329edeb7c186a137f9386828ca7307cd3557e3fea7225d3403fd6b")
    assert torch.equal(_noise_for(first), _noise_for(second))
    assert torch.equal(torch.random.get_rng_state(), global_state)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (("scene", "another-scene"), ("episode", 8), ("call", 3)),
)
def test_different_call_key_produces_different_seed(field, replacement):
    kwargs = {field: replacement}
    assert _metadata(**kwargs)["per_call_seed"] != _metadata()["per_call_seed"]


def test_unrelated_arm_calls_do_not_change_noise_for_same_key():
    target = _metadata(scene="shared-scene", episode=11, call=4)

    arm_a_noise = _noise_for(target)
    # The other arm may take a different number/order of RPC calls. Each one
    # consumes only its own local generator, so the shared key remains equal.
    for call in (0, 1, 7, 3, 12):
        _noise_for(_metadata(scene="other-path", episode=99, call=call))
    arm_b_noise = _noise_for(target)

    assert torch.equal(arm_a_noise, arm_b_noise)


def test_legacy_missing_metadata_is_accepted_only_when_not_required():
    assert validate_rpc_sampling_metadata(None, require_deterministic=False) is None
    with pytest.raises(ValueError, match="metadata is required"):
        validate_rpc_sampling_metadata(None, require_deterministic=True)


@pytest.mark.parametrize(
    "missing_field",
    (
        "sampling_protocol",
        "protocol_seed",
        "scene_id",
        "episode_id",
        "system2_call_index",
        "per_call_seed",
        "seed_sha256",
    ),
)
def test_partial_metadata_fails_closed(missing_field):
    metadata = _metadata()
    metadata.pop(missing_field)
    with pytest.raises(ValueError, match="metadata is incomplete"):
        validate_rpc_sampling_metadata(metadata, require_deterministic=False)


def test_tampered_seed_fails_server_side_sha256_rederivation():
    metadata = copy.deepcopy(_metadata())
    metadata["per_call_seed"] += 1

    with pytest.raises(ValueError, match="SHA256 rederivation"):
        validate_rpc_sampling_metadata(metadata, require_deterministic=True)


def test_progress_resume_contract_accepts_exact_deterministic_record():
    expected = build_rpc_progress_sampling_contract(
        protocol_seed=42,
        require_deterministic_sampling=True,
    )
    assert expected["rpc_protocol"] == HEATMAPVLN_RPC_PROTOCOL_VERSION
    validate_rpc_progress_sampling_contract(dict(expected), expected=expected)


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("rpc_protocol", "legacy-v1"),
        ("rpc_sampling_protocol", "other"),
        ("rpc_deterministic_sampling_enabled", False),
        ("rpc_protocol_seed", 43),
        ("rpc_require_deterministic_sampling", False),
    ),
)
def test_progress_resume_contract_rejects_mixed_sampling_rows(field, replacement):
    expected = build_rpc_progress_sampling_contract(
        protocol_seed=42,
        require_deterministic_sampling=True,
    )
    row = dict(expected)
    row[field] = replacement
    with pytest.raises(ValueError, match="progress sampling contract mismatch"):
        validate_rpc_progress_sampling_contract(row, expected=expected)

import pytest
import torch
from torch import nn

from scripts.tools.derive_heatmap_fullhead_init import derive_full_head_state
from scripts.tools.derive_heatmap_spatial_reset_init import (
    derive_spatial_reset_state,
)
from scripts.training.heatmap_warmstart import (
    FULL_HEAD_POLICY,
    SPATIAL_RESET_POLICY,
    validate_heatmap_warmstart_contract,
    verify_heatmap_warmstart_loaded,
)


class _ToyPipeline(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.qwen2_5_vl = nn.Module()
        self.qwen2_5_vl.lora_A = nn.Parameter(torch.ones(2, 2))
        self.heatmap_vln = nn.Module()
        self.heatmap_vln.llm_dpt_fusion = nn.Linear(2, 2, bias=False)
        self.heatmap_vln.coarse = nn.Linear(2, 2, bias=False)
        self.heatmap_vln.vit_dpt_fusion = nn.Linear(2, 2, bias=False)
        self.heatmap_vln.fine = nn.Module()
        self.heatmap_vln.fine.refine = nn.Sequential(
            nn.Linear(2, 2),
            nn.Identity(),
            nn.Linear(2, 2),
            nn.Identity(),
            nn.Linear(2, 1),
        )


def _stage_cfg():
    return {
        "trainable_modules": ["heatmap_vln"],
        "heatmap_warmstart_contract": {
            "policy": SPATIAL_RESET_POLICY,
            "expected_lora_tensors": 1,
            "expected_llm_dpt_tensors": 1,
            "expected_coarse_tensors": 1,
            "require_metadata": True,
        },
    }


def _metadata():
    return {
        "heatmap_warmstart_contract": {
            "policy": SPATIAL_RESET_POLICY,
            "kept_heatmap_modules": ["llm_dpt_fusion", "coarse"],
            "reset_heatmap_modules": ["vit_dpt_fusion", "fine"],
            "zero_initialized_parameters": [
                "heatmap_vln.fine.refine.4.weight",
                "heatmap_vln.fine.refine.4.bias",
            ],
        }
    }


def _full_head_stage_cfg():
    return {
        "trainable_modules": ["heatmap_vln"],
        "heatmap_warmstart_contract": {
            "policy": FULL_HEAD_POLICY,
            "expected_lora_tensors": 1,
            "expected_vit_dpt_tensors": 1,
            "expected_llm_dpt_tensors": 1,
            "expected_coarse_tensors": 1,
            "expected_fine_tensors": 6,
            "require_metadata": True,
        },
    }


def _full_head_metadata():
    return {
        "heatmap_warmstart_contract": {
            "policy": FULL_HEAD_POLICY,
            "kept_heatmap_modules": [
                "vit_dpt_fusion",
                "llm_dpt_fusion",
                "coarse",
                "fine",
            ],
        }
    }


def _full_source_state(model):
    return {
        name: value.detach().clone()
        for name, value in model.named_parameters()
    }


def test_derivation_omits_old_spatial_weights_and_zeros_fine_output():
    model = _ToyPipeline()
    derived = derive_spatial_reset_state(_full_source_state(model))

    assert len(derived) == 5
    assert not any("vit_dpt_fusion" in name for name in derived)
    assert not any("fine.refine.0" in name for name in derived)
    assert not any("fine.refine.2" in name for name in derived)
    assert torch.count_nonzero(
        derived["heatmap_vln.fine.refine.4.weight"]
    ) == 0
    assert torch.count_nonzero(
        derived["heatmap_vln.fine.refine.4.bias"]
    ) == 0


def test_full_head_derivation_preserves_complete_locator():
    model = _ToyPipeline()
    source = _full_source_state(model)
    derived = derive_full_head_state(source)

    assert set(derived) == set(source)
    assert all(torch.equal(derived[name], source[name]) for name in source)


def test_full_head_contract_accepts_complete_checkpoint():
    model = _ToyPipeline()
    state = derive_full_head_state(_full_source_state(model))
    report = validate_heatmap_warmstart_contract(
        model,
        state,
        _full_head_stage_cfg(),
        checkpoint_metadata=_full_head_metadata(),
        checkpoint_path="full-head.pth",
    )

    assert report["expected_loaded_tensors"] == 10
    assert report["counts"] == {
        "lora": 1,
        "vit_dpt_fusion": 1,
        "llm_dpt_fusion": 1,
        "coarse": 1,
        "fine": 6,
    }
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert missing == []
    assert unexpected == []
    verify_heatmap_warmstart_loaded(model, report, loaded_count=10)


def test_full_head_contract_rejects_missing_fine_tensor():
    model = _ToyPipeline()
    state = derive_full_head_state(_full_source_state(model))
    state.pop("heatmap_vln.fine.refine.4.bias")

    with pytest.raises(RuntimeError, match="tensor-count contract"):
        validate_heatmap_warmstart_contract(
            model,
            state,
            _full_head_stage_cfg(),
            checkpoint_metadata=_full_head_metadata(),
        )


def test_contract_accepts_exact_partial_checkpoint_and_postload_zero():
    model = _ToyPipeline()
    state = derive_spatial_reset_state(_full_source_state(model))
    report = validate_heatmap_warmstart_contract(
        model,
        state,
        _stage_cfg(),
        checkpoint_metadata=_metadata(),
        checkpoint_path="partial.pth",
    )

    assert report["expected_loaded_tensors"] == 5
    missing, unexpected = model.load_state_dict(state, strict=False)
    assert unexpected == []
    assert "heatmap_vln.vit_dpt_fusion.weight" in missing
    verify_heatmap_warmstart_loaded(model, report, loaded_count=5)


@pytest.mark.parametrize(
    "mutate, message",
    [
        (
            lambda state, model: state.update(
                {
                    "heatmap_vln.vit_dpt_fusion.weight":
                    model.heatmap_vln.vit_dpt_fusion.weight.detach().clone()
                }
            ),
            "forbidden tensors",
        ),
        (
            lambda state, model: state[
                "heatmap_vln.fine.refine.4.weight"
            ].fill_(1),
            "exactly zero",
        ),
        (
            lambda state, model: state.pop(
                "heatmap_vln.llm_dpt_fusion.weight"
            ),
            "tensor-count contract",
        ),
    ],
)
def test_contract_fails_closed_on_semantic_mismatch(mutate, message):
    model = _ToyPipeline()
    state = derive_spatial_reset_state(_full_source_state(model))
    mutate(state, model)

    with pytest.raises(RuntimeError, match=message):
        validate_heatmap_warmstart_contract(
            model,
            state,
            _stage_cfg(),
            checkpoint_metadata=_metadata(),
        )


def test_contract_requires_matching_metadata():
    model = _ToyPipeline()
    state = derive_spatial_reset_state(_full_source_state(model))

    with pytest.raises(RuntimeError, match="lacks required contract metadata"):
        validate_heatmap_warmstart_contract(
            model,
            state,
            _stage_cfg(),
            checkpoint_metadata={},
        )


def test_postload_guard_rejects_nonzero_fine_output():
    model = _ToyPipeline()
    state = derive_spatial_reset_state(_full_source_state(model))
    report = validate_heatmap_warmstart_contract(
        model,
        state,
        _stage_cfg(),
        checkpoint_metadata=_metadata(),
    )
    model.load_state_dict(state, strict=False)
    with torch.no_grad():
        model.heatmap_vln.fine.refine[4].bias.fill_(1)

    with pytest.raises(RuntimeError, match="zero fine residual"):
        verify_heatmap_warmstart_loaded(model, report, loaded_count=5)

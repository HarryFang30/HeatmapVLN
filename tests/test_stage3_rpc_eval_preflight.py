from pathlib import Path

import pytest
import torch

from scripts.evaluation.preflight_stage3_rpc_eval import (
    validate_base_checkpoint,
    validate_stage3_checkpoint,
    validate_stage3_config,
)


def _config(*, dim: int = 8, adapter_hidden_dim: int = 4) -> dict:
    return {
        "data": {
            "trajectory": {
                "panoramic_vlm_input": True,
                "structured_pano_output": True,
            }
        },
        "model": {
            "llm": {
                "hidden_dim": dim,
                "use_lora": True,
                "lora_rank": 32,
                "lora_layer_indices": list(range(28)),
                "lora_target_modules": ["q_proj", "k_proj", "v_proj", "o_proj"],
            },
            "action_head": {
                "nextdit": {
                    "enabled": True,
                    "pano_latent_adapter": {
                        "enabled": True,
                        "hidden_dim": adapter_hidden_dim,
                    },
                }
            },
        },
        "training": {
            "stages": [
                {
                    "name": "stage3",
                    "strict_trainable_modules": True,
                    "requires_base_checkpoint": True,
                    "require_complete_internnav_system1": True,
                    "base_checkpoint_lora_only": True,
                    "trainable_modules": ["pano_latent_adapter"],
                }
            ]
        },
    }


def _base_state() -> dict[str, torch.Tensor]:
    state = {}
    for layer in range(28):
        for module in ("q_proj", "k_proj", "v_proj", "o_proj"):
            prefix = (
                "qwen2_5_vl.model.base_model.model.model."
                f"layers.{layer}.self_attn.{module}"
            )
            state[f"{prefix}.lora_A.default.weight"] = torch.zeros(32, 8)
            state[f"{prefix}.lora_B.default.weight"] = torch.zeros(8, 32)
    return state


def _write_base(path: Path) -> None:
    torch.save(
        {
            "stage_name": "stage1_s2_panoramic_sft",
            "epoch": 5,
            "trainable_state_dict": _base_state(),
        },
        path,
    )


def _write_stage3(path: Path, base_path: Path, *, nonfinite: bool = False) -> None:
    cfg = _config()
    cfg["runtime"] = {"base_checkpoint": str(base_path.resolve())}
    state = {
        "pano_latent_adapter.mlp.0.weight": torch.zeros(4, 8),
        "pano_latent_adapter.mlp.0.bias": torch.zeros(4),
        "pano_latent_adapter.mlp.3.weight": torch.zeros(8, 4),
        "pano_latent_adapter.mlp.3.bias": torch.zeros(8),
    }
    if nonfinite:
        state["pano_latent_adapter.mlp.3.bias"][0] = float("nan")
    torch.save(
        {
            "stage_name": "stage3",
            "epoch": 2,
            "batch": None,
            "config": cfg,
            "trainable_state_dict": state,
        },
        path,
    )


def test_stage3_rpc_preflight_accepts_exact_all_layer_checkpoints(tmp_path):
    base_path = tmp_path / "base.pth"
    stage3_path = tmp_path / "stage3.pth"
    _write_base(base_path)
    _write_stage3(stage3_path, base_path)

    summary = validate_stage3_config(_config(), expected_adapter_hidden_dim=4)
    base = validate_base_checkpoint(base_path, summary)
    stage3 = validate_stage3_checkpoint(
        stage3_path,
        expected_epoch=2,
        expected_base_checkpoint=base_path,
        config_summary=summary,
    )

    assert base["lora_tensors"] == 224
    assert stage3["adapter_tensors"] == 4
    assert stage3["adapter_parameters"] == 76


def test_stage3_rpc_preflight_rejects_partial_lora_checkpoint(tmp_path):
    base_path = tmp_path / "base.pth"
    state = _base_state()
    state.pop(next(iter(state)))
    torch.save({"trainable_state_dict": state}, base_path)
    summary = validate_stage3_config(_config(), expected_adapter_hidden_dim=4)

    with pytest.raises(ValueError, match="Base LoRA checkpoint validation failed"):
        validate_base_checkpoint(base_path, summary)


def test_stage3_rpc_preflight_rejects_nonfinite_adapter(tmp_path):
    base_path = tmp_path / "base.pth"
    stage3_path = tmp_path / "stage3.pth"
    _write_base(base_path)
    _write_stage3(stage3_path, base_path, nonfinite=True)
    summary = validate_stage3_config(_config(), expected_adapter_hidden_dim=4)

    with pytest.raises(ValueError, match="non-finite"):
        validate_stage3_checkpoint(
            stage3_path,
            expected_epoch=2,
            expected_base_checkpoint=base_path,
            config_summary=summary,
        )

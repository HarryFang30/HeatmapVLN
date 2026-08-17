from src.config_schema import HeatmapWarmstartContractConfig


def test_full_head_warmstart_contract_is_accepted_by_config_schema():
    contract = HeatmapWarmstartContractConfig.model_validate(
        {
            "policy": "full_head_v1",
            "expected_lora_tensors": 224,
            "expected_vit_dpt_tensors": 12,
            "expected_llm_dpt_tensors": 10,
            "expected_coarse_tensors": 37,
            "expected_fine_tensors": 6,
            "require_metadata": True,
        }
    )

    assert contract.policy == "full_head_v1"
    assert contract.expected_vit_dpt_tensors == 12
    assert contract.expected_fine_tensors == 6

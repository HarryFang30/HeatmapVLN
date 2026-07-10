import torch
import torch.nn as nn

from src.models.qwen2_5_vl.integration import (
    Qwen2_5VLConfig,
    Qwen2_5VLIntegration,
)


class _MergedBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.proj = nn.Linear(2, 2)


class _FakePeftBackbone(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.lora_A = nn.Parameter(torch.ones(2, 1))
        self.base = nn.Linear(2, 2)
        self.merge_args = None

    def merge_and_unload(self, *, progressbar, safe_merge):
        self.merge_args = (progressbar, safe_merge)
        return _MergedBackbone()


def test_merge_lora_for_frozen_forward_removes_adapter_tensors():
    integration = Qwen2_5VLIntegration(
        Qwen2_5VLConfig(device='cpu', use_lora=True)
    )
    peft_model = _FakePeftBackbone()
    integration.model = peft_model
    integration._model_loaded = True

    merged_count = integration.merge_lora_for_frozen_forward(safe_merge=True)

    assert merged_count == 1
    assert peft_model.merge_args == (False, True)
    assert integration.config.use_lora is False
    assert not integration.model.training
    assert not any(param.requires_grad for param in integration.model.parameters())
    assert not any('lora_' in name for name, _ in integration.model.named_parameters())


def test_inference_tensor_clone_can_feed_trainable_adapter():
    with torch.inference_mode():
        frozen_hidden = torch.randn(2, 4, 8)
    materialized = frozen_hidden.clone()
    adapter = nn.Linear(8, 8)

    adapter(materialized).sum().backward()

    assert frozen_hidden.is_inference()
    assert not materialized.is_inference()
    assert adapter.weight.grad is not None

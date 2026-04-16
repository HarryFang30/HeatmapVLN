"""Runtime behavior regressions for config/AMP/model wiring."""

import sys
import types

import torch
from scripts.training.utils import (
    make_autocast_context,
    make_grad_scaler,
    resolve_amp_dtype,
)

from src.models import pipeline as pipeline_module


class DummyQwenIntegration:
    def __init__(self, config):
        self.config = config
        self._model_loaded = False


class DummyNextDiTActionConfig:
    def __init__(self, **kwargs):
        self.kwargs = kwargs


class DummyNextDiTActionHead(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg


def _install_dummy_action_module(monkeypatch):
    dummy_module = types.ModuleType("src.models.action")
    dummy_module.NextDiTActionConfig = DummyNextDiTActionConfig
    dummy_module.NextDiTActionHead = DummyNextDiTActionHead
    monkeypatch.setitem(sys.modules, "src.models.action", dummy_module)


class TestAmpHelpers:
    def test_resolve_amp_dtype(self):
        assert resolve_amp_dtype("bf16") == torch.bfloat16
        assert resolve_amp_dtype("fp16") == torch.float16
        assert resolve_amp_dtype("off") is None

    def test_make_grad_scaler_disabled_on_cpu(self):
        assert make_grad_scaler(torch.device("cpu"), "fp16") is None

    def test_make_autocast_context_is_noop_on_cpu(self):
        x = torch.ones(2, dtype=torch.float32)
        with make_autocast_context(torch.device("cpu"), "bf16"):
            y = x + 1
        assert y.dtype == torch.float32


class TestActionHeadToggle:
    def test_pipeline_skips_action_head_when_disabled(self, monkeypatch):
        monkeypatch.setattr(pipeline_module, "Qwen2_5VLIntegration", DummyQwenIntegration)
        _install_dummy_action_module(monkeypatch)

        model = pipeline_module.VLNPipeline(
            pipeline_module.VLNPipelineConfig(
                device="cpu",
                nextdit_enabled=True,
                enable_action_head=False,
            )
        )

        assert model.nextdit_action_head is None
        assert model.latent_queries is None

    def test_pipeline_builds_action_head_when_enabled(self, monkeypatch):
        monkeypatch.setattr(pipeline_module, "Qwen2_5VLIntegration", DummyQwenIntegration)
        _install_dummy_action_module(monkeypatch)

        model = pipeline_module.VLNPipeline(
            pipeline_module.VLNPipelineConfig(
                device="cpu",
                nextdit_enabled=True,
                enable_action_head=True,
            )
        )

        assert isinstance(model.nextdit_action_head, DummyNextDiTActionHead)
        assert model.latent_queries is not None

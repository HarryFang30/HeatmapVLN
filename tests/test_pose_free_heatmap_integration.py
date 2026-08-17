from types import SimpleNamespace

import pytest
import torch
import torch.nn as nn
from scripts.training.model_builder import set_trainable_modules
from scripts.training.optimizer import build_optimizer

from src.models import pipeline as pipeline_module
from src.models.heatmap.heatmap_vln import HeatmapVLN


class _FakeBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.visual = nn.Module()
        self.visual.blocks = nn.ModuleList([nn.Identity()])
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([nn.Identity()])


def _build_heatmap(*, decoder_mode=None, trajectory_config=None, heatmap_trains_backbone=False):
    kwargs = {}
    if decoder_mode is not None:
        kwargs["decoder_mode"] = decoder_mode
    return HeatmapVLN(
        qwen_model=_FakeBackbone(),
        processor=object(),
        c_vit=6,
        c_llm=8,
        c_fused=4,
        vit_layer_indices=[0],
        llm_layer_indices=[0],
        trajectory_config=trajectory_config,
        heatmap_trains_backbone=heatmap_trains_backbone,
        pose_free_config={
            "match_dim": 4,
            "visibility_hidden_dim": 3,
            "heatmap_size": (10, 12),
        },
        **kwargs,
    )


def _compact_features():
    torch.manual_seed(23)
    current_patches = {0: torch.randn(2, 4, 8, 8, 8)}
    history_queries = [
        [torch.randn(8, requires_grad=True) for _ in range(4)],
        [torch.randn(8, requires_grad=True) for _ in range(3)],
    ]
    return current_patches, history_queries


def test_compact_pose_free_decode_has_batched_contract_and_true_length_mask():
    heatmap = _build_heatmap(
        decoder_mode="pose_free_matcher",
        heatmap_trains_backbone=True,
    )
    llm_features, history_queries = _compact_features()

    assert heatmap.vit_layer_indices == []
    assert heatmap.feat_extractor.vit_layer_indices == []
    assert not heatmap.qwen.visual.blocks[0]._forward_hooks
    assert heatmap.vit_dpt_fusion is None
    assert heatmap.llm_dpt_fusion is None
    assert heatmap.coarse is None
    assert heatmap.fine is None

    output = heatmap._decode_feature_tensors_batch(
        vit_layer_tensors={},
        llm_layer_tensors=llm_features,
        history_queries_batch=history_queries,
        num_histories=[4, 3],
        device=torch.device("cpu"),
    )

    assert output["heatmaps"].shape == (2, 4, 4, 10, 12)
    assert output["visibility"].shape == (2, 4, 4)
    assert output["history_mask"].tolist() == [
        [True, True, True, True],
        [True, True, True, False],
    ]
    assert torch.count_nonzero(output["heatmaps"][1, 3]) == 0
    assert torch.count_nonzero(output["visibility"][1, 3]) == 0

    output["heatmap_logits"].square().mean().backward()
    for query in history_queries[0] + history_queries[1]:
        assert query.grad is not None
        assert query.grad.abs().sum() > 0


def test_pose_free_decode_fails_closed_on_relative_pose():
    heatmap = _build_heatmap(decoder_mode="pose_free_matcher")
    llm_features, history_queries = _compact_features()

    with pytest.raises(ValueError, match="fails closed"):
        heatmap._decode_feature_tensors_batch(
            vit_layer_tensors={},
            llm_layer_tensors=llm_features,
            history_queries_batch=history_queries,
            num_histories=[4, 3],
            device=torch.device("cpu"),
            history_rel_poses=torch.randn(2, 4, 4),
        )


def test_pose_free_constructor_rejects_enabled_trajectory_branch():
    with pytest.raises(ValueError, match="forbidden"):
        _build_heatmap(
            decoder_mode="pose_free_matcher",
            trajectory_config={"enable": True},
        )


def test_default_constructor_is_explicit_legacy_with_identical_state_layout():
    torch.manual_seed(31)
    default_model = _build_heatmap()
    torch.manual_seed(31)
    explicit_model = _build_heatmap(decoder_mode="legacy")

    assert default_model.decoder_mode == "legacy"
    assert default_model.pose_free_matcher is None
    assert default_model.vit_dpt_fusion is not None
    assert default_model.llm_dpt_fusion is not None
    assert default_model.coarse is not None
    assert default_model.fine is not None
    assert default_model.state_dict().keys() == explicit_model.state_dict().keys()
    for key, value in default_model.state_dict().items():
        torch.testing.assert_close(value, explicit_model.state_dict()[key], rtol=0, atol=0)


def test_training_policy_and_optimizer_select_only_pose_free_head():
    class _PipelineStub(nn.Module):
        def __init__(self):
            super().__init__()
            self.heatmap_vln = _build_heatmap(decoder_mode="pose_free_matcher")

    class _LoggerStub:
        def info(self, *_args, **_kwargs):
            pass

        def warning(self, *_args, **_kwargs):
            pass

    pipeline = _PipelineStub()
    stage_cfg = {"trainable_modules": ["heatmap_vln"]}
    set_trainable_modules(pipeline, stage_cfg, _LoggerStub())

    matcher_params = list(pipeline.heatmap_vln.pose_free_matcher.parameters())
    assert matcher_params
    assert all(parameter.requires_grad for parameter in matcher_params)
    assert all(
        not parameter.requires_grad
        for name, parameter in pipeline.named_parameters()
        if not name.startswith("heatmap_vln.pose_free_matcher.")
    )

    optimizer = build_optimizer(
        pipeline,
        cfg={"optim": {"heatmap_lr": 2e-4, "weight_decay": 1e-2}},
        stage_cfg=stage_cfg,
    )
    optimized_ids = {id(parameter) for group in optimizer.param_groups for parameter in group["params"]}
    assert optimized_ids == {id(parameter) for parameter in matcher_params}
    assert all(group["name"].startswith("heatmap_pose_free_matcher") for group in optimizer.param_groups)


def test_pipeline_pose_free_materialization_preserves_empty_vit_hook_list(monkeypatch):
    captured = {}

    class _HeatmapStub(nn.Module):
        def __init__(self, **kwargs):
            super().__init__()
            captured.update(kwargs)
            self.vit_dpt_fusion = None
            self.llm_dpt_fusion = None
            self.fine = None
            self.coarse = None
            self.pose_free_matcher = nn.Linear(2, 2)

    monkeypatch.setattr(pipeline_module, "HeatmapVLN", _HeatmapStub)
    pipeline = pipeline_module.VLNPipeline.__new__(pipeline_module.VLNPipeline)
    nn.Module.__init__(pipeline)
    pipeline.heatmap_vln = None
    pipeline._heatmap_enabled = True
    pipeline.device = torch.device("cpu")
    pipeline.qwen2_5_vl = SimpleNamespace(
        _model_loaded=True,
        model=nn.Identity(),
        processor=object(),
    )
    pipeline.config = SimpleNamespace(
        heatmap_vit_layer_indices=[],
        heatmap_llm_layer_indices=[0],
        heatmap_decoder_mode="pose_free_matcher",
        heatmap_trajectory_config=None,
        heatmap_pose_free_config={"match_dim": 2},
        heatmap_size=(10, 12),
        heatmap_c_vit=6,
        heatmap_c_llm=8,
        heatmap_c_fused=4,
        heatmap_trains_backbone=True,
        enable_runtime_timing=False,
        dtype=torch.float32,
    )

    pipeline._ensure_heatmap_vln()

    assert captured["vit_layer_indices"] == []
    assert captured["decoder_mode"] == "pose_free_matcher"
    assert captured["pose_free_config"]["heatmap_size"] == (10, 12)

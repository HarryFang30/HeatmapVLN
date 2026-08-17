from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.evaluation.heatmap import materialize_and_load_heatmap_checkpoint
from scripts.training.collate import collate_fn
from scripts.tools.diagnose_heatmap_shortcuts import binary_curves, heatmap_head_state_dict
from src.data.panoramic_tokenized_collator import PanoramicTokenizedCollator
from src.data.sliding_window_dataset import VLNSlidingWindowDataset
from src.models.heatmap.feature_extractor import FeatureExtractor
from src.models.heatmap.input_constructor import VIEW_NAMES, construct_input
from src.models.qwen2_5_vl.integration import Qwen2_5VLIntegration


def _panorama(base: int) -> dict[str, Image.Image]:
    return {
        name: Image.new("RGB", (2, 2), color=(base + index, 0, 0))
        for index, name in enumerate(VIEW_NAMES)
    }


def _image_colors(messages: list[dict]) -> list[int]:
    content = messages[0]["content"]
    return [item["image"].getpixel((0, 0))[0] for item in content if item["type"] == "image"]


def test_heatmap_layout_places_current_occurrences_first_and_anchors_after_history_images():
    messages = construct_input(
        current_views=_panorama(10),
        history_panoramas=[_panorama(20), _panorama(30)],
        heatmap_layout=True,
    )
    assert _image_colors(messages) == [10, 11, 12, 13, 20, 21, 22, 23, 30, 31, 32, 33]

    content = messages[0]["content"]
    anchor_indices = [
        index
        for index, item in enumerate(content)
        if item["type"] == "text" and item["text"].startswith("Historical observation")
    ]
    assert len(anchor_indices) == 2
    for anchor_index in anchor_indices:
        assert [item["type"] for item in content[anchor_index - 4:anchor_index]] == ["image"] * 4


def test_navigation_layout_remains_history_then_current():
    messages = construct_input(
        current_views=_panorama(10),
        history_panoramas=[_panorama(20)],
    )
    assert _image_colors(messages) == [20, 21, 22, 23, 10, 11, 12, 13]


def _sample(num_histories: int) -> dict:
    return {
        "history_frames": torch.zeros(1, 3, 2, 2),
        "current_frame": torch.zeros(3, 2, 2),
        "heatmap": torch.zeros(num_histories, 4, 2, 2),
        "action": torch.zeros(2),
        "action_valid": 0.0,
        "discrete_action": 1,
        "is_stop": 0.0,
        "text": "",
        "current_views": torch.zeros(4, 3, 2, 2),
        "history_panoramas": torch.zeros(num_histories, 4, 3, 2, 2),
        "gt_visibility": torch.zeros(num_histories, 4),
        "history_rel_poses": torch.zeros(num_histories, 4),
    }


def test_standard_collate_mask_uses_real_heatmap_history_length():
    output = collate_fn([_sample(3), _sample(2)])
    assert output["history_frames"].shape[1] == 1
    assert output["history_mask"].tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]


def test_tokenized_collator_mask_uses_real_heatmap_history_length():
    output = PanoramicTokenizedCollator._stack_padded_history_frames([_sample(3), _sample(2)])
    assert output["history_frames"].shape[1] == 1
    assert output["history_mask"].tolist() == [[1.0, 1.0, 1.0], [1.0, 1.0, 0.0]]


def test_declared_meter_depth_takes_precedence_over_first_frame_heuristic(tmp_path):
    clip = tmp_path / "scene" / "clip_000001"
    clip.mkdir(parents=True)
    (clip / "meta.json").write_text(
        json.dumps({"data_format": {"depth_unit": "meters"}}),
        encoding="utf-8",
    )
    dataset = object.__new__(VLNSlidingWindowDataset)
    dataset.load_depth = True
    dataset.clips = [clip]
    assert dataset._detect_depth_format() is True


def test_append_only_collection_can_be_restricted_by_global_clip_id(tmp_path):
    for scene in ("scene_a", "scene_b"):
        for clip_id in (1, 2000, 2001, 6000):
            (tmp_path / scene / f"clip_{clip_id:06d}").mkdir(parents=True)
    dataset = object.__new__(VLNSlidingWindowDataset)
    dataset.root = tmp_path
    dataset.split = "all"
    dataset.max_clips = 0
    dataset.max_clip_id = 2000
    clips = dataset._enumerate_clips()
    assert len(clips) == 4
    assert all(VLNSlidingWindowDataset._numeric_clip_id(clip) <= 2000 for clip in clips)


def test_head_checkpoint_excludes_shared_qwen_reference():
    class DummyHeatmap(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qwen = torch.nn.Linear(2, 2)
            self.head = torch.nn.Linear(2, 1)

    state = heatmap_head_state_dict(DummyHeatmap())
    assert state
    assert all(not key.startswith("qwen.") for key in state)
    assert {key for key in state} == {"head.weight", "head.bias"}


def test_binary_curves_are_order_invariant_for_tied_scores():
    scores = np.asarray([0.5, 0.5, 0.5, 0.5])
    labels = np.asarray([1, 0, 1, 0])
    auroc, auprc = binary_curves(scores, labels)
    reversed_auroc, reversed_auprc = binary_curves(scores[::-1], labels[::-1])
    assert auroc == reversed_auroc == 0.5
    assert auprc == reversed_auprc == 0.5


def _bare_feature_extractor() -> FeatureExtractor:
    extractor = object.__new__(FeatureExtractor)
    extractor.vit_features = {}
    extractor.llm_hidden_states = {}
    extractor._batch_capture_plan = None
    extractor._captured_batch_vit = {}
    extractor._captured_batch_llm = {}
    extractor._captured_batch_queries = None
    extractor._capture_suspend_depth = 0
    extractor.detach_features = False
    return extractor


def test_feature_capture_suspension_is_reentrant_and_preserves_normal_hooks():
    extractor = _bare_feature_extractor()
    vit_hook = extractor._make_vit_hook(7)
    llm_hook = extractor._make_llm_hook(13)
    vit_output = torch.randn(4, 3)
    llm_output = torch.randn(1, 4, 3)

    vit_hook(None, None, vit_output)
    llm_hook(None, None, (llm_output,))
    assert extractor.vit_features[7] is vit_output
    assert extractor.llm_hidden_states[13] is llm_output

    with extractor.suspend_capture():
        assert extractor.vit_features == {}
        assert extractor.llm_hidden_states == {}
        vit_hook(None, None, vit_output)
        llm_hook(None, None, llm_output)
        assert extractor.vit_features == {}
        assert extractor.llm_hidden_states == {}

        with extractor.suspend_capture():
            vit_hook(None, None, vit_output)
            assert extractor.vit_features == {}

        llm_hook(None, None, llm_output)
        assert extractor.llm_hidden_states == {}

    assert extractor.vit_features == {}
    assert extractor.llm_hidden_states == {}
    vit_hook(None, None, vit_output)
    llm_hook(None, None, llm_output)
    assert extractor.vit_features[7] is vit_output
    assert extractor.llm_hidden_states[13] is llm_output


def test_feature_capture_suspension_clears_and_recovers_after_exception():
    extractor = _bare_feature_extractor()
    vit_hook = extractor._make_vit_hook(7)
    vit_output = torch.randn(4, 3)

    with pytest.raises(RuntimeError, match="rehearsal failed"):
        with extractor.suspend_capture():
            extractor.vit_features[99] = torch.ones(1)
            raise RuntimeError("rehearsal failed")

    assert extractor._capture_suspend_depth == 0
    assert extractor.vit_features == {}
    assert extractor.llm_hidden_states == {}
    vit_hook(None, None, vit_output)
    assert extractor.vit_features[7] is vit_output


def test_single_panorama_decode_receives_relative_pose():
    integration = object.__new__(Qwen2_5VLIntegration)
    torch.nn.Module.__init__(integration)
    integration.device = torch.device("cpu")
    integration.config = SimpleNamespace(heatmap_trains_backbone=False)
    integration._forward_model_inputs = lambda *args, **kwargs: (None, None, 0, None, None)

    captured = {}

    class FakeHeatmap:
        feat_extractor = SimpleNamespace(clear=lambda: None)

        def prepare_qwen_inputs(self, **kwargs):
            return {"input_ids": torch.ones(1, 1, dtype=torch.long)}, 2

        def decode_from_inputs(self, inputs, num_history, history_rel_poses=None):
            captured["poses"] = history_rel_poses
            return {"heatmaps": torch.empty(0)}

    poses = torch.randn(2, 4)
    integration._forward_single_panorama(
        current_views=torch.zeros(4, 3, 2, 2),
        history_panoramas=torch.zeros(2, 4, 3, 2, 2),
        return_hidden_states=False,
        heatmap_vln=FakeHeatmap(),
        history_rel_poses=poses,
    )
    assert captured["poses"] is poses


def test_heatmap_evaluator_materializes_lazy_modules_before_loading(tmp_path):
    class LazyQwen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._model_loaded = False

        def _load_model(self):
            if not self._model_loaded:
                self.model = torch.nn.Module()
                self.model.register_parameter("lora_A", torch.nn.Parameter(torch.zeros(2, 2)))
                self._model_loaded = True

    class DummyHeatmap(torch.nn.Module):
        def __init__(self, qwen):
            super().__init__()
            self.qwen = qwen
            self.head = torch.nn.Linear(2, 1)

    class LazyPipeline(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qwen2_5_vl = LazyQwen()
            self.heatmap_vln = None

        def _ensure_heatmap_vln(self):
            if self.heatmap_vln is None:
                self.heatmap_vln = DummyHeatmap(self.qwen2_5_vl.model)

    checkpoint_path = tmp_path / "heatmap.pth"
    torch.save(
        {
            "trainable_state_dict": {
                "qwen2_5_vl.model.lora_A": torch.full((2, 2), 3.0),
                "heatmap_vln.head.weight": torch.full((1, 2), 4.0),
                "heatmap_vln.head.bias": torch.full((1,), 5.0),
            }
        },
        checkpoint_path,
    )
    model = LazyPipeline()
    result = materialize_and_load_heatmap_checkpoint(model, str(checkpoint_path))
    assert result["matched_lora_tensors"] == 1
    assert result["matched_heatmap_head_tensors"] == 2
    assert torch.equal(model.qwen2_5_vl.model.lora_A, torch.full((2, 2), 3.0))
    assert torch.equal(model.heatmap_vln.head.weight, torch.full((1, 2), 4.0))
    assert torch.equal(model.heatmap_vln.head.bias, torch.full((1,), 5.0))


def _legacy_heatmap_pipeline():
    class LazyQwen(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self._model_loaded = False

        def _load_model(self):
            if not self._model_loaded:
                self.model = torch.nn.Module()
                self.model.register_parameter(
                    "lora_A",
                    torch.nn.Parameter(torch.zeros(2, 2)),
                )
                self._model_loaded = True

    class DummyHeatmap(torch.nn.Module):
        def __init__(self, qwen):
            super().__init__()
            self.qwen = qwen
            self.head = torch.nn.Linear(2, 1)

    class LazyPipeline(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.qwen2_5_vl = LazyQwen()
            self.heatmap_vln = None

        def _ensure_heatmap_vln(self):
            if self.heatmap_vln is None:
                self.heatmap_vln = DummyHeatmap(self.qwen2_5_vl.model)

    return LazyPipeline()


def test_heatmap_evaluator_strictly_composes_legacy_head_with_recorded_base(
    tmp_path,
):
    base_path = tmp_path / "base.pth"
    torch.save(
        {
            "trainable_state_dict": {
                "qwen2_5_vl.model.lora_A": torch.full((2, 2), 3.0),
                # A base head must never overwrite the newly trained head.
                "heatmap_vln.head.weight": torch.full((1, 2), -4.0),
                "heatmap_vln.head.bias": torch.full((1,), -5.0),
            }
        },
        base_path,
    )
    checkpoint_path = tmp_path / "legacy_heatmap.pth"
    torch.save(
        {
            "stage_idx": 0,
            "stage_name": "heatmap",
            "config": {
                "training": {
                    "stages": [
                        {
                            "name": "heatmap",
                            "requires_base_checkpoint": True,
                        }
                    ]
                },
                "runtime": {"base_checkpoint": str(base_path)},
            },
            "trainable_state_dict": {
                "heatmap_vln.head.weight": torch.full((1, 2), 4.0),
                "heatmap_vln.head.bias": torch.full((1,), 5.0),
            },
        },
        checkpoint_path,
    )

    model = _legacy_heatmap_pipeline()
    result = materialize_and_load_heatmap_checkpoint(
        model,
        str(checkpoint_path),
    )

    assert result["matched_lora_tensors"] == 1
    assert result["matched_heatmap_head_tensors"] == 2
    assert result["composed_base_lora_tensors"] == 1
    assert torch.equal(
        model.qwen2_5_vl.model.lora_A,
        torch.full((2, 2), 3.0),
    )
    assert torch.equal(
        model.heatmap_vln.head.weight,
        torch.full((1, 2), 4.0),
    )
    assert torch.equal(
        model.heatmap_vln.head.bias,
        torch.full((1,), 5.0),
    )


def test_heatmap_evaluator_refuses_ambiguous_legacy_base_path(tmp_path):
    checkpoint_path = tmp_path / "legacy_heatmap.pth"
    torch.save(
        {
            "stage_idx": 0,
            "stage_name": "heatmap",
            "config": {
                "training": {
                    "stages": [
                        {
                            "name": "heatmap",
                            "requires_base_checkpoint": True,
                        }
                    ]
                },
                "runtime": {"base_checkpoint": "relative/base.pth"},
            },
            "trainable_state_dict": {
                "heatmap_vln.head.weight": torch.full((1, 2), 4.0),
                "heatmap_vln.head.bias": torch.full((1,), 5.0),
            },
        },
        checkpoint_path,
    )

    with pytest.raises(RuntimeError, match="non-absolute base path"):
        materialize_and_load_heatmap_checkpoint(
            _legacy_heatmap_pipeline(),
            str(checkpoint_path),
        )

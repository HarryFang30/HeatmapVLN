from __future__ import annotations

import random
from collections import Counter
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import scripts.tools.train_heatmap_joint_pilot as pilot
import torch
from scripts.tools.summarize_task4_pilot import generation_coverage
from scripts.tools.train_heatmap_joint_pilot import (
    DeterministicEpochBatchStream,
    _generation_summary,
    all_indices_from_scenes,
    build_sft_dataset_and_collator,
    configure_trainable_parameters,
    effective_milestone_steps,
    exact_sft_sample,
    generic_selection_contract,
    gradient_conflict,
    milestone_evaluation_argv,
    rehearsal_stream_contract,
    requested_milestone_steps,
    select_indices_from_scenes,
    select_scene_partition,
    sft_correct_label_logprobs,
    sft_dataset_contract,
    sft_forward_loss,
)
from torch import nn

from src.data.panoramic_tokenized_collator import IGNORE_INDEX


class _LoRAProjection(nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = nn.Parameter(torch.ones(2, 2))
        self.lora_B = nn.Parameter(torch.ones(2, 2))


class _LoRABackbone(nn.Module):
    def __init__(self, layers: int):
        super().__init__()
        self.model = nn.Module()
        self.model.layers = nn.ModuleList([_LoRAProjection() for _ in range(layers)])


class _HeatmapHead(nn.Module):
    def __init__(self, qwen: nn.Module):
        super().__init__()
        self.qwen = qwen
        self.decoder = nn.Linear(2, 1)


class _PilotModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen2_5_vl = _LoRABackbone(4)
        self.heatmap_vln = _HeatmapHead(self.qwen2_5_vl)
        self.unrelated = nn.Linear(2, 2)


def test_trainable_scope_is_head_plus_reachable_lora_only():
    model = _PilotModel()
    head, lora, _groups = configure_trainable_parameters(
        model,
        "joint-rehearsal",
        max_lora_layer=2,
    )
    assert len(head) == 2
    assert len(lora) == 6
    assert all(parameter.requires_grad for parameter in head)
    assert all(parameter.requires_grad for parameter in lora.values())
    assert all(".layers.3." not in name for name in lora)
    assert not any(parameter.requires_grad for parameter in model.unrelated.parameters())
    assert not any(parameter.requires_grad for name, parameter in model.named_parameters() if ".layers.3." in name)


def test_gradient_conflict_recovers_accumulated_rehearsal_gradient():
    parameter = nn.Parameter(torch.zeros(2))
    heatmap = torch.tensor([1.0, 0.0])
    rehearsal = torch.tensor([-1.0, 1.0])
    parameter.grad = heatmap + rehearsal
    result = gradient_conflict({"p": heatmap}, {"p": parameter})
    assert result["cosine"] == pytest.approx(-(2**-0.5))
    assert result["heatmap_norm"] == 1.0
    assert result["weighted_rehearsal_norm"] == pytest.approx(2**0.5)
    assert result["negative_tensor_fraction"] == 1.0


def test_explicit_preservation_gradient_telemetry_and_accumulation():
    parameter = nn.Parameter(torch.zeros(2))
    parameter.grad = torch.tensor([1.0, 0.0])
    parameters = {"model.layers.0.q_proj.lora_A.weight": parameter}
    rehearsal = (torch.tensor([-1.0, 1.0]),)

    conflict, layers, norm = pilot.explicit_rehearsal_gradient_telemetry(
        parameters,
        rehearsal,
    )
    assert conflict["cosine"] == pytest.approx(-(2**-0.5))
    assert norm == pytest.approx(2**0.5)
    assert layers["0"]["nonzero_tensors"] == 1
    pilot.accumulate_explicit_gradients(parameters, rehearsal)
    torch.testing.assert_close(parameter.grad, torch.tensor([0.0, 1.0]))


class _SceneDataset:
    def __init__(self, root: Path):
        self.root = root
        self.clips = [
            root / "scene_a" / "clip_1",
            root / "scene_b" / "clip_2",
            root / "scene_c" / "clip_3",
        ]
        self.sample_index = [(0, 5), (0, 6), (1, 5), (1, 6), (2, 5), (2, 6)]
        self._clip_valid_frames = {0: [5, 6, 7], 1: [5, 6, 7], 2: [5, 6, 7]}


def test_scene_holdout_and_round_robin_selection_are_deterministic(tmp_path):
    dataset = _SceneDataset(tmp_path)
    rehearsal, holdout = select_scene_partition(
        dataset,
        seed=42,
        holdout_scene_count=1,
    )
    repeated = select_scene_partition(dataset, seed=42, holdout_scene_count=1)
    assert (rehearsal, holdout) == repeated
    assert set(rehearsal).isdisjoint(holdout)
    assert set(rehearsal) | set(holdout) == {"scene_a", "scene_b", "scene_c"}

    indices = select_indices_from_scenes(dataset, rehearsal, limit=4)
    selected_scenes = {dataset.clips[dataset.sample_index[index][0]].parent.name for index in indices}
    assert selected_scenes <= set(rehearsal)
    assert len(indices) == 4
    clip_contract = sft_dataset_contract(dataset)
    assert clip_contract["clip_count"] == 3
    assert clip_contract["scene_count"] == 3
    assert clip_contract["balanced_view_manifest"] is None


def test_sft_selection_keeps_one_terminal_stop_and_deduplicates_oversampling(tmp_path):
    dataset = _SceneDataset(tmp_path)
    dataset.sample_index = [
        (0, 5),
        (0, 7),
        (0, 7),
        (0, 7),
        (0, 7),
    ]
    indices = select_indices_from_scenes(dataset, ["scene_a"], limit=10)
    assert indices == [0, 1]
    assert [dataset.sample_index[index] for index in indices] == [(0, 5), (0, 7)]


def test_sft_selection_uses_fixed_pixel_stop_quota_and_records_counts(tmp_path):
    dataset = _SceneDataset(tmp_path)
    dataset.sample_index = [
        (0, 5),
        (0, 6),
        (0, 7),
        (0, 7),
        (1, 5),
        (1, 6),
        (1, 7),
        (1, 7),
        (2, 5),
        (2, 6),
        (2, 7),
        (2, 7),
    ]
    indices = select_indices_from_scenes(
        dataset,
        ["scene_a", "scene_b", "scene_c"],
        limit=8,
    )
    contract = generic_selection_contract(dataset, indices)
    assert contract["category_counts"] == {"pixel": 6, "stop": 2}
    assert len(contract["sample_identities"]) == len(set(contract["sample_identities"]))


def test_full_sft_candidate_pool_preserves_source_stop_oversampling(tmp_path):
    dataset = _SceneDataset(tmp_path)
    dataset.sample_index = [
        (0, 5),
        (0, 7),
        (0, 7),
        (0, 7),
        (1, 5),
    ]
    indices = all_indices_from_scenes(dataset, ["scene_a"])
    assert indices == [0, 1, 2, 3]
    contract = generic_selection_contract(dataset, indices)
    assert contract["category_counts"] == {"pixel": 1, "stop": 3}
    assert contract["sample_count"] == 4
    assert contract["unique_physical_sample_count"] == 2
    assert contract["duplicate_physical_sample_count"] == 2


def test_rehearsal_stream_is_deterministic_and_without_replacement_per_epoch(tmp_path):
    stream = DeterministicEpochBatchStream(range(5), batch_size=2, seed=46)
    batches = stream.planned_batches(4)
    repeated = DeterministicEpochBatchStream(range(5), batch_size=2, seed=46).planned_batches(4)
    assert batches == repeated
    assert [len(batch.indices) for batch in batches] == [2, 2, 1, 2]
    assert [batch.epoch for batch in batches] == [0, 0, 0, 1]
    assert sorted(index for batch in batches[:3] for index in batch.indices) == list(range(5))
    assert len({index for batch in batches[:3] for index in batch.indices}) == 5

    dataset = _SceneDataset(tmp_path)
    contract = rehearsal_stream_contract(
        dataset,
        list(range(len(dataset.sample_index))),
        batch_size=2,
        seed=46,
        train_steps=2,
    )
    repeated_contract = rehearsal_stream_contract(
        dataset,
        list(range(len(dataset.sample_index))),
        batch_size=2,
        seed=46,
        train_steps=2,
    )
    assert contract == repeated_contract
    assert contract["no_replacement_within_epoch"]
    assert contract["planned_sample_count"] == 4


class _PooledSFTCollator:
    def __call__(self, samples):
        assert len(samples) == 2
        return {
            "current_frame": torch.zeros(2, 3, 2, 2),
            "pano_inputs": {
                "labels": torch.tensor(
                    [
                        [IGNORE_INDEX, 1, 1, IGNORE_INDEX],
                        [IGNORE_INDEX, 3, IGNORE_INDEX, IGNORE_INDEX],
                    ]
                ),
            },
            "pano_num_histories": [0, 0],
        }


class _PooledSFTModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))

    def forward(self, **kwargs):
        assert kwargs["video_frames"].shape[:2] == (2, 1)
        labels = kwargs["panoramic_inputs"]["labels"][:, 1:]
        valid = labels != IGNORE_INDEX
        # This deliberately differs from the mean of the two per-sample means:
        # pooled=(1+1+3)/3, per-sample=(1+3)/2.
        return {"lm_loss": labels[valid].float().mean() * self.scale}


def test_sft_forward_loss_collates_once_and_uses_batch_token_pooling():
    model = _PooledSFTModel()
    loss, label_tokens, sample_label_tokens = sft_forward_loss(
        model,
        _PooledSFTCollator(),
        [{"sample": 1}, {"sample": 2}],
        torch.device("cpu"),
    )
    assert label_tokens == 3
    assert sample_label_tokens == [2, 1]
    assert loss.item() == pytest.approx(5 / 3)
    loss.backward()
    assert model.scale.grad.item() == pytest.approx(5 / 3)


class _CorrectLogprobSFTModel(nn.Module):
    def __init__(self, *, consume_rng: bool = False):
        super().__init__()
        self.scale = nn.Parameter(torch.tensor(1.0))
        self.consume_rng = consume_rng

    def forward(self, **kwargs):
        if self.consume_rng:
            random.random()
            np.random.rand()
            torch.rand(1)
        labels = kwargs["panoramic_inputs"]["labels"]
        semantic = pilot.expected_correct_label_alignment(labels)
        correct_tokens = torch.tensor(
            [token for row in semantic["sample_correct_token_ids"] for token in row],
            dtype=torch.float32,
        )
        logprobs = -correct_tokens * self.scale
        union = sorted({position for row in semantic["sample_predictor_positions"] for position in row})
        return {
            "lm_correct_label_logprobs": logprobs,
            "lm_correct_label_alignment": {
                **semantic,
                "backend": "hf_logits_to_keep_tensor_predictor_union_v1",
                "predictor_position_union": union,
                "returned_logits_shape": [labels.shape[0], len(union), 10],
                "returned_logprob_dtype": "torch.float32",
            },
        }


def test_correct_label_logprob_wrapper_checks_alignment_and_keeps_gradient():
    model = _CorrectLogprobSFTModel()
    values, label_tokens, sample_tokens, alignment = sft_correct_label_logprobs(
        model,
        _PooledSFTCollator(),
        [{"sample": 1}, {"sample": 2}],
        torch.device("cpu"),
    )
    torch.testing.assert_close(values, torch.tensor([-1.0, -1.0, -3.0]))
    assert label_tokens == 3
    assert sample_tokens == [2, 1]
    assert alignment["sample_predictor_positions"] == [[0, 1], [0]]
    assert len(alignment["alignment_sha256"]) == 64
    values.sum().backward()
    assert model.scale.grad.item() == -5.0


class _CaptureExtractor(nn.Module):
    @contextmanager
    def suspend_capture(self):
        yield


class _TeacherCacheModel(_CorrectLogprobSFTModel):
    def __init__(self):
        super().__init__(consume_rng=True)
        self.qwen2_5_vl = nn.Module()
        self.heatmap_vln = nn.Module()
        self.heatmap_vln.feat_extractor = _CaptureExtractor()


class _TeacherCacheDataset:
    def __init__(self, root: Path):
        self.root = root
        self.clips = [root / "scene_a" / f"clip_{index}" for index in range(4)]
        self.sample_index = [(index, 5) for index in range(4)]

    def _build_sample(self, index):
        return {"index": index, "valid": True}

    @staticmethod
    def _result_has_system2_sft_target(sample):
        return sample["valid"]


def test_teacher_cache_covers_exact_batches_and_restores_rng_and_modes(tmp_path):
    pilot.set_seed(1234)
    model = _TeacherCacheModel()
    model.train()
    model.heatmap_vln.eval()
    modes_before = pilot.module_mode_contract(model)
    rng_before = pilot.rng_state_sha256(pilot.capture_rng_state())
    planned = [
        pilot.StreamBatch(epoch=0, start_position=0, indices=(2, 0)),
        pilot.StreamBatch(epoch=0, start_position=2, indices=(3, 1)),
    ]
    dataset = _TeacherCacheDataset(tmp_path / "data")

    records, contract = pilot.precompute_teacher_correct_logprob_cache(
        model=model,
        dataset=dataset,
        collator=_PooledSFTCollator(),
        planned_batches=planned,
        device=torch.device("cpu"),
        output_path=tmp_path / "teacher.pt",
        source_lora_sha256="initial-lora-hash",
    )

    assert [record["dataset_indices"] for record in records] == [[2, 0], [3, 1]]
    assert all(record["teacher_logprobs"].dtype == torch.float32 for record in records)
    assert all(record["teacher_logprobs"].device.type == "cpu" for record in records)
    assert contract["record_count"] == 2
    assert contract["planned_sample_count"] == 4
    assert contract["rng_restored_exactly"]
    assert contract["module_modes_restored_exactly"]
    assert len(contract["cache_sha256"]) == 64
    assert len(contract["artifact_file_sha256"]) == 64
    assert pilot.rng_state_sha256(pilot.capture_rng_state()) == rng_before
    assert pilot.module_mode_contract(model) == modes_before
    student, _tokens, _per_sample, _alignment = sft_correct_label_logprobs(
        model,
        _PooledSFTCollator(),
        [dataset._build_sample(2), dataset._build_sample(0)],
        torch.device("cpu"),
    )
    assert torch.equal(student.detach(), records[0]["teacher_logprobs"])
    assert float(
        (student.float() - records[0]["teacher_logprobs"]).square().mean().item()
    ) == 0.0

    _records_again, repeated = pilot.precompute_teacher_correct_logprob_cache(
        model=model,
        dataset=dataset,
        collator=_PooledSFTCollator(),
        planned_batches=planned,
        device=torch.device("cpu"),
        output_path=tmp_path / "teacher-again.pt",
        source_lora_sha256="initial-lora-hash",
    )
    assert repeated["cache_sha256"] == contract["cache_sha256"]


def test_milestone_steps_are_sorted_deduplicated_and_train_bounded():
    assert requested_milestone_steps("100,0,25,25, 50") == [0, 25, 50, 100]
    assert effective_milestone_steps("100,0,25,25,50", 49) == [0, 25]
    assert effective_milestone_steps("", 100) == []
    with pytest.raises(ValueError, match="non-negative"):
        requested_milestone_steps("-1")


def test_milestone_evaluation_uses_fresh_eval_only_process(tmp_path):
    args = SimpleNamespace(
        config=tmp_path / "heatmap.yaml",
        data_root=tmp_path / "heatmaps",
        sft_config=tmp_path / "sft.yaml",
        sft_data_root=tmp_path / "sft",
        device="cuda:0",
        num_history=2,
        max_clip_id=2000,
        heatmap_train_samples=128,
        heatmap_val_samples=64,
        sft_train_samples=0,
        sft_batch_size=4,
        sft_val_samples=64,
        sft_generation_samples=64,
        sft_holdout_scenes=7,
        sft_max_clips=0,
        head_learning_rate=1e-4,
        lora_learning_rate=1e-4,
        rehearsal_objective="hard-ce",
        rehearsal_weight=1.0,
        weight_decay=1e-2,
        grad_clip=1.0,
        max_trainable_lora_layer=20,
        gradient_cosine_every=25,
        log_every=10,
        seed=42,
        max_new_tokens=16,
        coord_tolerance=15.0,
        interventions="blank-images",
        skip_generation=False,
    )
    checkpoint = tmp_path / "checkpoint_step_000025.pth"
    argv = milestone_evaluation_argv(
        args,
        checkpoint,
        tmp_path / "eval-step-25",
    )
    assert argv[argv.index("--mode") + 1] == "head-only"
    assert argv[argv.index("--train-steps") + 1] == "0"
    assert argv[argv.index("--checkpoint") + 1] == str(checkpoint.resolve())
    assert argv[argv.index("--head-checkpoint") + 1] == str(checkpoint.resolve())
    assert argv[argv.index("--milestone-steps") + 1] == ""
    assert argv[argv.index("--sft-batch-size") + 1] == "4"
    assert argv[argv.index("--rehearsal-objective") + 1] == "hard-ce"


class _ExactSFTDataset:
    def __init__(self, *, valid: bool = True):
        self.valid = valid
        self.built = []

    def __getitem__(self, _index):
        raise AssertionError("fallback-enabled __getitem__ must not be used")

    def _build_sample(self, index):
        self.built.append(index)
        return {"index": index, "valid": self.valid}

    @staticmethod
    def _result_has_system2_sft_target(sample):
        return sample["valid"]


def test_exact_sft_retrieval_bypasses_fallback_and_validates_target():
    dataset = _ExactSFTDataset()
    assert exact_sft_sample(dataset, 7) == {"index": 7, "valid": True}
    assert dataset.built == [7]

    with pytest.raises(RuntimeError, match="no valid System2 target"):
        exact_sft_sample(_ExactSFTDataset(valid=False), 3)


def test_sft_dataset_build_disables_unused_history_heatmaps(monkeypatch, tmp_path):
    captured = {}
    dataset = object()

    def fake_build_dataset(_cfg, **kwargs):
        captured.update(kwargs)
        return dataset

    monkeypatch.setattr(
        pilot,
        "load_training_config",
        lambda _path: {
            "data": {"root": "unused", "trajectory": {}},
            "training": {"stages": [{}]},
            "model": {"llm": {}},
        },
    )
    monkeypatch.setattr(pilot, "build_training_dataset", fake_build_dataset)
    tokenizer = SimpleNamespace(padding_side=None, truncation_side=None)
    model = SimpleNamespace(
        qwen2_5_vl=SimpleNamespace(
            processor=SimpleNamespace(tokenizer=tokenizer),
        ),
    )
    args = SimpleNamespace(
        sft_config="config.yaml",
        sft_data_root=str(tmp_path),
        sft_max_clips=0,
    )
    built, _collator, _cfg = build_sft_dataset_and_collator(args, model)
    assert built is dataset
    assert captured["load_history_heatmap"] is False


def test_generation_summary_marks_missing_attempts_as_incomplete():
    metrics = Counter(attempted=3, total=2, errors=1)
    summary = _generation_summary(metrics, requested_samples=3)
    assert summary["requested_samples"] == 3
    assert summary["attempted_samples"] == 3
    assert summary["samples"] == 2
    assert summary["errors"] == 1
    assert not summary["complete_coverage"]


def test_summary_generation_coverage_requires_both_error_free_phases():
    complete = {
        "samples": 4,
        "requested_samples": 4,
        "attempted_samples": 4,
        "errors": 0,
        "skipped_no_target": 0,
        "complete_coverage": True,
    }
    report = {
        "sft_retention": {
            "generation_before": dict(complete),
            "generation_after": dict(complete),
        },
    }
    assert generation_coverage(report)["complete"]

    report["sft_retention"]["generation_after"]["errors"] = 1
    assert not generation_coverage(report)["complete"]

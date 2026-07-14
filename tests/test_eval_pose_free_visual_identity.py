from __future__ import annotations

import copy
from types import SimpleNamespace

import pytest
import scripts.tools.eval_pose_free_visual_identity as evaluator
import torch


class _Adapter(torch.nn.Module):
    def __init__(self, value: float = 1.0):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.tensor([value]))


class _TinyModel(torch.nn.Module):
    def __init__(self, lora_value: float = 1.0):
        super().__init__()
        self.qwen2_5_vl = torch.nn.Module()
        self.qwen2_5_vl.model = torch.nn.Module()
        self.qwen2_5_vl.model.layers = torch.nn.ModuleList([_Adapter(lora_value)])
        self.heatmap_vln = torch.nn.Module()
        self.heatmap_vln.pose_free_matcher = torch.nn.Linear(2, 1, bias=False)


def _args(**overrides):
    values = {
        "cell": "warmup-original",
        "warmup_checkpoint": "warmup.pth",
        "trained_checkpoint": None,
        "paired_checkpoint": None,
        "selection_split": "val",
        "standard_only": False,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _contracts(model: torch.nn.Module):
    lora_hash = evaluator.base_pilot.tensor_state_sha256(evaluator.base_pilot.lora_state_dict(model))
    stage1 = {
        "path": "/pinned/stage1.pth",
        "file_sha256": "stage1-file",
        "matched_lora_tensors": 1,
        "loaded_lora_sha256": lora_hash,
    }
    manifest = {
        "manifest_sha256": "manifest",
        "file_sha256": "manifest-file",
        "source_inventory_sha256": "inventory",
        "max_clip_id": 2000,
        "source_inventory_clips": 2000,
        "num_history": 4,
        "train_identity_sha256": "train",
        "val_identity_sha256": "val",
        "train_samples": 128,
        "val_samples": 40,
        "scene_disjoint": True,
        "split_source_inventories": {
            "train": {"inventory_sha256": "inventory", "clips": 2000},
            "val": {"inventory_sha256": "inventory", "clips": 2000},
        },
        "minimum_target_separation_pixels": 12.0,
        "identity_targets_per_sample": 4,
    }
    config = {
        "history_query_source": evaluator.visual_pilot.HISTORY_QUERY_SOURCE,
        "qwen_forward_batch_size": 1,
        "qwen_forwards_per_sample": 4,
    }
    runtime = {
        "history_query_source": evaluator.visual_pilot.HISTORY_QUERY_SOURCE,
        "qwen_forward_batch_size": 1,
        "qwen_forwards_per_sample": 4,
    }
    return stage1, manifest, config, runtime


def _optimization_contract(train_mode: str) -> dict:
    lora_mode = train_mode != "head-warmup"
    return {
        "optimizer": {
            "name": "AdamW",
            "betas": list(evaluator.visual_pilot.ADAMW_BETAS),
            "eps": evaluator.visual_pilot.ADAMW_EPS,
            "weight_decay": 0.01,
            "amsgrad": False,
        },
        "train_steps": 2,
        "seed": 42,
        "learning_rates": {
            "head": 1e-4,
            "lora": 1e-4,
            "active_group": "reachable_lora" if lora_mode else "pose_free_matcher_warmup",
            "active": 1e-4,
        },
        "grad_clip": 1.0,
        "max_trainable_lora_layer": 20,
        "gradient_checkpointing": lora_mode,
        "protocol_reachable_lora_tensors": 1,
        "protocol_reachable_lora_layers": [0],
        "expected_trainable_lora_tensors": 1 if lora_mode else 0,
        "actual_trainable_lora_tensors": 1 if lora_mode else 0,
        "expected_trainable_lora_layers": [0] if lora_mode else [],
        "actual_trainable_lora_layers": [0] if lora_mode else [],
        "actual_trainable_head_tensors": 0 if lora_mode else 1,
        "qwen_forward_batch_size": 1,
        "qwen_forwards_per_sample": 4,
    }


def _patch_tiny_protocol(monkeypatch) -> None:
    monkeypatch.setattr(evaluator.visual_pilot, "EXPECTED_LORA_TENSORS", 1)
    monkeypatch.setattr(evaluator.visual_pilot, "EXPECTED_TRAINABLE_LORA_TENSORS", 1)
    monkeypatch.setattr(evaluator.visual_pilot, "EXPECTED_TRAINABLE_LORA_LAYERS", (0,))


def _warmup_payload(model, stage1, manifest, config, runtime):
    head = evaluator.base_pilot.pose_free_head_state_dict(model)
    lora = evaluator.base_pilot.lora_state_dict(model)
    return {
        "schema": evaluator.visual_pilot.CHECKPOINT_SCHEMA,
        "protocol": evaluator.visual_pilot.PROTOCOL,
        "train_mode": "head-warmup",
        "step": 2,
        "training_pid": 123,
        "head_state_dict": head,
        "lora_state_dict": lora,
        "head_state_sha256": evaluator.base_pilot.tensor_state_sha256(head),
        "lora_state_sha256": evaluator.base_pilot.tensor_state_sha256(lora),
        "initial_head_sha256": "random-head",
        "initial_lora_sha256": stage1["loaded_lora_sha256"],
        "expected_lora_tensors": 1,
        "stage1_s2_contract": stage1,
        "manifest_contract": manifest,
        "pose_free_config_contract": config,
        "runtime_contract": runtime,
        "warmup_checkpoint_contract": None,
        "training_sample_schedule_sha256": "shared-schedule",
        "optimization_contract": _optimization_contract("head-warmup"),
        "loss_contract": evaluator.visual_pilot.expected_loss_contract("head-warmup"),
    }


def _trained_payload(
    warmup_payload,
    warmup_path,
    *,
    train_mode: str,
    lora_value: float,
):
    payload = copy.deepcopy(warmup_payload)
    trained_model = _TinyModel(lora_value)
    lora = evaluator.base_pilot.lora_state_dict(trained_model)
    payload.update(
        {
            "train_mode": train_mode,
            "lora_state_dict": lora,
            "lora_state_sha256": evaluator.base_pilot.tensor_state_sha256(lora),
            "initial_head_sha256": warmup_payload["head_state_sha256"],
            "warmup_checkpoint_contract": {
                "schema": evaluator.visual_pilot.CHECKPOINT_SCHEMA,
                "protocol": evaluator.visual_pilot.PROTOCOL,
                "path": str(warmup_path.resolve()),
                "file_sha256": evaluator.base_pilot.file_sha256(warmup_path),
                "head_state_sha256": warmup_payload["head_state_sha256"],
                "lora_state_sha256": warmup_payload["lora_state_sha256"],
                "step": warmup_payload["step"],
                "training_sample_schedule_sha256": warmup_payload["training_sample_schedule_sha256"],
                "optimization_contract": warmup_payload["optimization_contract"],
            },
            "optimization_contract": _optimization_contract(train_mode),
            "loss_contract": evaluator.visual_pilot.expected_loss_contract(train_mode),
        }
    )
    return payload


def test_cli_cells_require_exact_checkpoint_sources():
    evaluator.validate_args(_args())
    with pytest.raises(ValueError, match="forbids"):
        evaluator.validate_args(_args(trained_checkpoint="identity.pth"))
    for cell in ("identity-trained", "heatmap-control-trained"):
        with pytest.raises(ValueError, match="requires --trained-checkpoint"):
            evaluator.validate_args(_args(cell=cell))
        evaluator.validate_args(
            _args(
                cell=cell,
                trained_checkpoint=f"{cell}.pth",
                paired_checkpoint="counterpart.pth",
            )
        )
    with pytest.raises(ValueError, match="paths must differ"):
        evaluator.validate_args(
            _args(
                cell="identity-trained",
                trained_checkpoint="warmup.pth",
                paired_checkpoint="control.pth",
            )
        )
    with pytest.raises(ValueError, match="requires --standard-only"):
        evaluator.validate_args(_args(selection_split="train"))
    evaluator.validate_args(_args(selection_split="train", standard_only=True))


@pytest.mark.parametrize(
    ("cell", "train_mode", "lora_value"),
    (
        ("identity-trained", "lora-identity", 2.0),
        ("heatmap-control-trained", "lora-heatmap-control", 3.0),
    ),
)
def test_trained_cell_loads_shared_warmup_head_and_only_selected_lora(
    tmp_path,
    monkeypatch,
    cell,
    train_mode,
    lora_value,
):
    _patch_tiny_protocol(monkeypatch)
    source = _TinyModel(1.0)
    with torch.no_grad():
        source.heatmap_vln.pose_free_matcher.weight.copy_(torch.tensor([[7.0, -4.0]]))
    stage1, manifest, config, runtime = _contracts(source)
    warmup_payload = _warmup_payload(source, stage1, manifest, config, runtime)
    warmup_path = tmp_path / "warmup.pth"
    torch.save(warmup_payload, warmup_path)
    trained_payload = _trained_payload(
        warmup_payload,
        warmup_path,
        train_mode=train_mode,
        lora_value=lora_value,
    )
    trained_path = tmp_path / "trained.pth"
    torch.save(trained_payload, trained_path)
    counterpart_mode = "lora-heatmap-control" if train_mode == "lora-identity" else "lora-identity"
    counterpart_payload = _trained_payload(
        warmup_payload,
        warmup_path,
        train_mode=counterpart_mode,
        lora_value=lora_value + 10.0,
    )
    counterpart_path = tmp_path / "counterpart.pth"
    torch.save(counterpart_payload, counterpart_path)

    target = _TinyModel(1.0)
    selected, sources = evaluator.load_eval_cell_strict(
        target,
        cell=cell,
        warmup_checkpoint=warmup_path,
        trained_checkpoint=trained_path,
        paired_checkpoint=counterpart_path,
        stage1_contract=stage1,
        manifest_contract=manifest,
        config_contract=config,
        runtime_contract=runtime,
    )

    torch.testing.assert_close(
        target.heatmap_vln.pose_free_matcher.weight,
        source.heatmap_vln.pose_free_matcher.weight,
    )
    assert target.qwen2_5_vl.model.layers[0].lora_A.item() == lora_value
    assert selected["fresh_stage1_lora_loaded_before_cell_state"] is True
    assert selected["active_head_sha256"] == warmup_payload["head_state_sha256"]
    assert selected["active_lora_sha256"] == trained_payload["lora_state_sha256"]
    assert sources["head"]["source"] == "shared-head-warmup"
    assert sources["lora"]["source"] == train_mode
    assert selected["identity_control_pair_gate"]["passed"] is True


def test_warmup_original_keeps_fresh_stage1_lora(tmp_path, monkeypatch):
    _patch_tiny_protocol(monkeypatch)
    source = _TinyModel(1.0)
    with torch.no_grad():
        source.heatmap_vln.pose_free_matcher.weight.fill_(5.0)
    stage1, manifest, config, runtime = _contracts(source)
    warmup_payload = _warmup_payload(source, stage1, manifest, config, runtime)
    warmup_path = tmp_path / "warmup.pth"
    torch.save(warmup_payload, warmup_path)
    target = _TinyModel(1.0)

    selected, sources = evaluator.load_eval_cell_strict(
        target,
        cell="warmup-original",
        warmup_checkpoint=warmup_path,
        trained_checkpoint=None,
        paired_checkpoint=None,
        stage1_contract=stage1,
        manifest_contract=manifest,
        config_contract=config,
        runtime_contract=runtime,
    )

    assert selected["active_lora_sha256"] == stage1["loaded_lora_sha256"]
    assert target.qwen2_5_vl.model.layers[0].lora_A.item() == 1.0
    assert sources["lora"]["source"] == "stage1-s2"


def test_trained_cell_rejects_a_different_warmup_file(tmp_path, monkeypatch):
    _patch_tiny_protocol(monkeypatch)
    source = _TinyModel(1.0)
    stage1, manifest, config, runtime = _contracts(source)
    warmup_payload = _warmup_payload(source, stage1, manifest, config, runtime)
    warmup_path = tmp_path / "warmup.pth"
    torch.save(warmup_payload, warmup_path)
    trained_payload = _trained_payload(
        warmup_payload,
        warmup_path,
        train_mode="lora-identity",
        lora_value=2.0,
    )
    trained_payload["warmup_checkpoint_contract"]["file_sha256"] = "different-warmup"
    trained_path = tmp_path / "identity.pth"
    torch.save(trained_payload, trained_path)

    with pytest.raises(RuntimeError, match=r"supplied warmup checkpoint.*file_sha256"):
        evaluator.load_eval_cell_strict(
            _TinyModel(1.0),
            cell="identity-trained",
            warmup_checkpoint=warmup_path,
            trained_checkpoint=trained_path,
            paired_checkpoint=tmp_path / "unused-control.pth",
            stage1_contract=stage1,
            manifest_contract=manifest,
            config_contract=config,
            runtime_contract=runtime,
        )


def _record(sample_id: str, offset: float = 0.0) -> dict:
    visibility = torch.arange(16, dtype=torch.float32).reshape(1, 4, 4) + offset
    heatmaps = torch.arange(4 * 4 * 2 * 2, dtype=torch.float32).reshape(1, 4, 4, 2, 2)
    heatmaps = heatmaps + offset
    gt_heatmaps = torch.zeros(4, 4, 2, 2)
    for history_slot in range(4):
        gt_heatmaps[history_slot, history_slot, 0, 0] = 1
    return {
        "sample_id": sample_id,
        "target_slot": None,
        "visibility": visibility,
        "heatmaps": heatmaps,
        "heatmap_logits": heatmaps.clone(),
        "gt_visibility": torch.eye(4),
        "gt_heatmaps": gt_heatmaps,
    }


def _swap_records(standard: list[dict]) -> list[dict]:
    result = []
    for base in standard:
        for target_slot in range(4):
            item = {key: value.clone() if torch.is_tensor(value) else value for key, value in base.items()}
            item["target_slot"] = target_slot
            item["visibility"][:, target_slot] += 100
            item["heatmaps"][:, target_slot] += 100
            item["heatmap_logits"][:, target_slot] += 100
            result.append(item)
    return result


def test_single_swap_gate_is_exact_for_every_untargeted_output():
    standard = [_record("a"), _record("b", 1000)]
    swaps = _swap_records(standard)

    gate = evaluator.assert_single_swap_untargeted_invariance(standard, swaps)

    assert gate["passed"] is True
    assert gate["swap_pairs"] == 8
    assert gate["untargeted_output_slots"] == 24
    assert gate["tensor_comparisons"] == 72

    swaps[0]["heatmaps"][:, 1, 0, 0, 0] += 1
    with pytest.raises(RuntimeError, match="changed an untargeted output"):
        evaluator.assert_single_swap_untargeted_invariance(standard, swaps)


def test_compact_record_preserves_target_grounded_score_identity():
    size = 8
    targets = ((0, 1, 1), (0, 2, 3), (0, 4, 5), (0, 6, 2))
    logits = torch.full((1, 4, 4, size, size), -4.0)
    gt_visibility = torch.zeros(4, 4)
    gt_heatmaps = torch.zeros(4, 4, size, size)
    for query, (view, y, x) in enumerate(targets):
        logits[0, query, 0, 1, 1] = 7.0
        logits[0, query, view, y, x] = 9.0
        gt_visibility[query, view] = 1.0
        gt_heatmaps[query, view, y, x] = 1.0
    bf16_probabilities = logits.to(torch.bfloat16).sigmoid().float()
    assert bf16_probabilities[0, 1, 0, 1, 1] == 1.0
    assert bf16_probabilities[0, 1, 0, 2, 3] == 1.0
    record = {
        "sample_id": "score-sample",
        "target_slot": None,
        "visibility": torch.where(gt_visibility.bool(), 8.0, -8.0).unsqueeze(0),
        "heatmaps": bf16_probabilities,
        "heatmap_logits": logits.to(torch.bfloat16).float(),
        "gt_visibility": gt_visibility,
        "gt_heatmaps": gt_heatmaps,
    }

    compact = evaluator.compact_visual_identity_record(record)
    scores = torch.tensor(compact["target_score_matrix"])

    assert scores.shape == (4, 4)
    assert scores.argmax(dim=-1).tolist() == [0, 1, 2, 3]
    assert torch.all(scores.diagonal() > scores.masked_fill(torch.eye(4).bool(), -torch.inf).max(dim=1).values)
    assert compact["score_reconstruction"] == {
        "source": "explicit_raw_heatmap_logits",
        "inverse": None,
        "raw_logits_opt_in": "return_heatmap_logits=True",
        "normalization": "per_view_spatial_log_softmax",
        "target_extraction": "primary_visible_gt_heatmap_peak",
        "target_sampling": "circular_panorama_bilinear_grid_sample_align_corners_false",
        "matrix_axes": ["history_query", "ground_truth_target"],
        "matrix_shape": [4, 4],
    }
    assert compact["probability_reconstructed_pred_xy"][1][0] == [1, 1]
    assert compact["pred_xy"][1][0] == [3, 2]
    assert compact["global_pred_view_xy"][1] == [0, 3, 2]
    assert compact["visibility_reconstruction"]["learned_readout_used"] is False
    assert compact["peak_reconstruction"]["bf16_sigmoid_probability_used"] is False


def test_raw_marginal_view_and_global_map_are_distinct_registered_readouts():
    height, width = 2, 3
    logits = torch.full((1, 4, 4, height, width), -10.0)
    # Broad moderate evidence wins the view marginal; a single sharper point
    # in another view wins global MAP.
    logits[:, :, 0] = 0.0
    logits[:, :, 1, 1, 2] = 1.0
    gt_visibility = torch.zeros(4, 4)
    gt_heatmaps = torch.zeros(4, 4, height, width)
    for history_slot in range(4):
        gt_visibility[history_slot, 0] = 1
        gt_heatmaps[history_slot, 0, 0, 0] = 1
    record = {
        "sample_id": "marginal-vs-map",
        "target_slot": None,
        "visibility": torch.zeros(1, 4, 4),
        "heatmaps": logits.sigmoid(),
        "heatmap_logits": logits,
        "gt_visibility": gt_visibility,
        "gt_heatmaps": gt_heatmaps,
    }

    compact = evaluator.compact_visual_identity_record(record)

    assert [max(range(4), key=row.__getitem__) for row in compact["visibility_logits"]] == [0, 0, 0, 0]
    assert compact["global_pred_view_xy"] == [[1, 2, 1]] * 4


def test_evaluation_emits_all_existing_metrics_and_fail_closed_gates(monkeypatch):
    standard = [_record("a"), _record("b", 1000)]
    permutation = [3, 2, 1, 0]
    shuffled = []
    for base in standard:
        item = {key: value.clone() if torch.is_tensor(value) else value for key, value in base.items()}
        item["history_permutation"] = permutation
        item["visibility"] = item["visibility"][:, permutation]
        item["heatmaps"] = item["heatmaps"][:, permutation]
        item["heatmap_logits"] = item["heatmap_logits"][:, permutation]
        shuffled.append(item)
    swaps = _swap_records(standard)
    calls = []

    def fake_evaluate(
        _model,
        _criterion,
        _dataset,
        intervention,
        _device,
        *,
        return_heatmap_logits,
    ):
        assert return_heatmap_logits is True
        calls.append(intervention)
        if intervention == "history-shuffle":
            records = shuffled
        elif intervention == "single-anchor-swap":
            records = swaps
        else:
            records = standard
        metrics = {"existing_metric": intervention, "samples": len(records), "loss": 0.0}
        if intervention == "blank-images":
            metrics["blank_input_identity_gate"] = {"passed": True, "bitwise_exact": True}
            metrics["blank_output_identity_gate"] = {"passed": True, "bitwise_exact": True}
        if intervention == "single-anchor-swap":
            metrics["source_samples"] = 2
            metrics["swap_evaluations_per_sample"] = 4
        return metrics, records

    monkeypatch.setattr(evaluator.base_pilot, "evaluate_intervention", fake_evaluate)
    monkeypatch.setattr(
        evaluator,
        "compact_visual_identity_record",
        lambda record: {"sample_id": record["sample_id"], "target_slot": record["target_slot"]},
    )

    evaluations, predictions, gates = evaluator.evaluate_all_interventions(
        object(),
        object(),
        [0, 1],
        torch.device("cpu"),
    )

    assert calls == list(evaluator.EVAL_INTERVENTIONS)
    assert set(evaluations) == set(evaluator.EVAL_INTERVENTIONS)
    assert evaluations["standard"]["legacy_evaluation"]["existing_metric"] == "standard"
    assert evaluations["standard"]["metric_source_contract"]["legacy_visibility_readout_used"] is False
    assert gates["history-shuffle"]["bitwise_exact"] is True
    assert gates["single-anchor-swap"]["bitwise_exact"] is True
    assert gates["blank-images"]["passed"] is True
    assert len(predictions["single-anchor-swap"]) == 8


def test_standard_only_generalization_diagnostic_is_explicit_and_raw_logit_backed(monkeypatch):
    standard = [_record("a"), _record("b", 1000)]
    calls = []

    def fake_evaluate(
        _model,
        _criterion,
        _dataset,
        intervention,
        _device,
        *,
        return_heatmap_logits,
    ):
        calls.append((intervention, return_heatmap_logits))
        return {"samples": len(standard), "loss": 0.0}, standard

    monkeypatch.setattr(evaluator.base_pilot, "evaluate_intervention", fake_evaluate)
    monkeypatch.setattr(
        evaluator,
        "compact_visual_identity_record",
        lambda record: {"sample_id": record["sample_id"]},
    )

    evaluations, predictions, gates = evaluator.evaluate_all_interventions(
        object(),
        object(),
        [0, 1],
        torch.device("cpu"),
        interventions=("standard",),
    )

    assert calls == [("standard", True)]
    assert set(evaluations) == {"standard"}
    assert set(predictions) == {"standard"}
    assert gates["standard"]["passed"] is True
    with pytest.raises(ValueError, match="full protocol or standard-only"):
        evaluator.evaluate_all_interventions(
            object(),
            object(),
            [0, 1],
            torch.device("cpu"),
            interventions=("current-shuffle",),
        )

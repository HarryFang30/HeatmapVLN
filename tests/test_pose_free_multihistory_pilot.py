from types import SimpleNamespace

import pytest
import scripts.tools.train_pose_free_multihistory_pilot as pilot_module
import torch
import yaml
from scripts.tools.train_pose_free_multihistory_pilot import (
    CHECKPOINT_SCHEMA,
    INTERVENTIONS,
    assert_blank_output_identity,
    assert_history_permutation_equivariance,
    compute_metrics,
    exact_sample,
    file_sha256,
    flatten_isolated_pair_chains,
    forward_loss,
    load_pilot_checkpoint_strict,
    load_pilot_config,
    paired_single_swap_output_change,
    pose_free_config_contract,
    regroup_isolated_pair_outputs,
    tensor_state_sha256,
    transform_sample,
    validate_args,
)

from src.models.heatmap import HeatmapVLNLoss


def _args(**overrides):
    values = {
        "phase": "train",
        "branch": "head-only",
        "pilot_checkpoint": None,
        "eval_head_checkpoint": None,
        "eval_lora": "trained",
        "train_steps": 100,
        "grad_clip": 1.0,
        "log_every": 10,
        "max_trainable_lora_layer": 20,
        "device": "cpu",
        "data_root": "/data",
        "model_path": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _sample(k=4):
    histories = torch.arange(k * 4 * 3 * 2 * 2, dtype=torch.float32).reshape(k, 4, 3, 2, 2)
    history_frames = histories[:, 0].clone()
    current_views = torch.full((4, 3, 2, 2), 1000.0)
    heatmaps = torch.zeros(k, 4, 8, 8)
    visibility = torch.zeros(k, 4)
    for slot in range(k):
        visibility[slot, slot] = 1
        heatmaps[slot, slot, 2 + slot, 1 + slot] = 1
    return {
        "current_views": current_views,
        "current_frame": current_views[0],
        "history_panoramas": histories,
        "history_frames": history_frames,
        "gt_visibility": visibility,
        "heatmap": heatmaps,
        "_task36c_audit": {
            "manifest_sample_id": "sample-a",
            "runtime_sample_id": "sample-a",
            "runtime_history_frames": list(range(k)),
            "current_frame": 20,
            "pose_inputs_removed": True,
        },
    }


def test_config_is_strict_pose_free_single_hook(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {"root": "old", "image_size": [224, 224], "init_hm_size": [64, 64]},
                "model": {
                    "device": "cuda",
                    "llm": {"model_path": "/model", "lora_dropout": 0.5},
                    "heatmap": {
                        "decoder_mode": "legacy",
                        "trajectory": {"enable": True},
                        "vit_layer_indices": [7, 15],
                        "llm_layer_indices": [6, 13, 20],
                    },
                    "action_head": {"enable": True},
                },
                "loss": {"heatmap_vln": {}},
            }
        ),
        encoding="utf-8",
    )
    args = _args(config=str(config_path), branch="heatmap-lora")

    cfg = load_pilot_config(args)
    contract = pose_free_config_contract(cfg)

    assert contract["decoder_mode"] == "pose_free_matcher"
    assert contract["trajectory_enabled"] is False
    assert contract["vit_layer_indices"] == []
    assert contract["llm_layer_indices"] == [20]
    assert contract["isolated_pair_chains"] is True
    assert contract["histories_per_qwen_chain"] == 1
    assert contract["qwen_forward_batch_size"] == 1
    assert contract["qwen_forwards_per_sample"] == 4
    assert cfg["model"]["llm"]["gradient_checkpointing"] is True
    assert cfg["model"]["action_head"]["enable"] is False


def test_cli_requires_phase_separated_checkpoint_contract():
    validate_args(_args())
    with pytest.raises(ValueError, match="evaluation-only"):
        validate_args(_args(pilot_checkpoint="pilot.pth"))
    with pytest.raises(ValueError, match="requires --pilot-checkpoint"):
        validate_args(_args(phase="eval"))
    validate_args(_args(phase="eval", pilot_checkpoint="pilot.pth"))
    with pytest.raises(ValueError, match="evaluation-only"):
        validate_args(_args(eval_head_checkpoint="head.pth"))
    with pytest.raises(ValueError, match="branch heatmap-lora"):
        validate_args(
            _args(
                phase="eval",
                pilot_checkpoint="pilot.pth",
                eval_head_checkpoint="head.pth",
            )
        )
    with pytest.raises(ValueError, match="eval-lora trained"):
        validate_args(
            _args(
                phase="eval",
                branch="heatmap-lora",
                pilot_checkpoint="pilot.pth",
                eval_head_checkpoint="head.pth",
                eval_lora="off",
            )
        )
    validate_args(
        _args(
            phase="eval",
            branch="heatmap-lora",
            pilot_checkpoint="pilot.pth",
            eval_head_checkpoint="head.pth",
        )
    )
    with pytest.raises(ValueError, match="layer at 20"):
        validate_args(_args(max_trainable_lora_layer=19))


def test_transform_interventions_never_create_pose_and_single_swap_is_local():
    sample = _sample()
    partner = _sample()
    partner["history_panoramas"] = partner["history_panoramas"] + 5000
    partner["history_frames"] = partner["history_frames"] + 5000
    partner["current_views"] = partner["current_views"] + 2000
    partner["current_frame"] = partner["current_frame"] + 2000

    standard = transform_sample(sample, intervention="standard")
    swapped = transform_sample(
        sample,
        intervention="single-anchor-swap",
        partner=partner,
        target_slot=2,
    )
    assert "history_rel_poses" not in standard
    assert torch.equal(swapped["history_panoramas"][:2], standard["history_panoramas"][:2])
    assert torch.equal(swapped["history_panoramas"][3], standard["history_panoramas"][3])
    assert torch.equal(swapped["history_panoramas"][2], partner["history_panoramas"][2])
    assert torch.equal(swapped["gt_heatmaps"], standard["gt_heatmaps"])
    assert swapped["metadata"]["target_slot"] == 2

    shuffled = transform_sample(sample, intervention="history-shuffle")
    assert torch.equal(shuffled["history_panoramas"], sample["history_panoramas"].flip(0))
    current = transform_sample(sample, intervention="current-shuffle", partner=partner)
    assert torch.equal(current["current_views"], partner["current_views"])
    blank = transform_sample(sample, intervention="blank-images")
    assert torch.count_nonzero(blank["current_views"]) == 0
    assert set(INTERVENTIONS) == {
        "standard",
        "blank-images",
        "history-shuffle",
        "current-shuffle",
        "single-anchor-swap",
    }


def test_flatten_and_regroup_use_four_independent_one_history_chains():
    transformed = transform_sample(_sample(), intervention="standard")
    chains = flatten_isolated_pair_chains(transformed)

    assert chains["video_frames"].shape == (4, 2, 3, 2, 2)
    assert chains["current_views"].shape == (4, 4, 3, 2, 2)
    assert chains["history_panoramas"].shape == (4, 1, 4, 3, 2, 2)
    assert chains["num_histories"] == [1, 1, 1, 1]
    for slot in range(4):
        torch.testing.assert_close(
            chains["history_panoramas"][slot, 0],
            transformed["history_panoramas"][slot],
        )
        torch.testing.assert_close(chains["current_views"][slot], transformed["current_views"])

    outputs = [
        {
            "visibility": torch.randn(1, 1, 4),
            "heatmaps": torch.randn(1, 1, 4, 8, 8),
        }
        for _ in range(4)
    ]
    visibility, heatmaps = regroup_isolated_pair_outputs(outputs, num_histories=4)
    assert visibility.shape == (1, 4, 4)
    assert heatmaps.shape == (1, 4, 4, 8, 8)
    torch.testing.assert_close(visibility[0], torch.cat([item["visibility"] for item in outputs])[:, 0])


def test_exact_sample_separates_manifest_identity_from_epoch_runtime_order():
    class FakeDataset:
        _sample_failure_count = 0
        _explicit_identities = ["scene/clip:current=20:history=1,2,3,4"]
        _explicit_canonical_frames = [(1, 2, 3, 4)]
        _explicit_history_frames = [(4, 2, 1, 3)]
        _explicit_records = [{"relative_clip": "scene/clip", "current_frame": 20}]
        sample_index = [(0, 20)]

        def __getitem__(self, _index):
            return {}

    sample = exact_sample(FakeDataset(), 0)

    assert sample["_task36c_audit"]["manifest_sample_id"].endswith("history=1,2,3,4")
    assert sample["_task36c_audit"]["runtime_sample_id"].endswith("history=4,2,1,3")
    assert not any(key.startswith("explicit_") for key in sample)


def test_forward_loss_rejects_pose_and_calls_four_isolated_b1_forwards():
    class FakeModel:
        def __init__(self):
            self.seen = []
            self.scale = torch.nn.Parameter(torch.tensor(0.0))

        def __call__(self, **kwargs):
            self.seen.append(kwargs)
            output = {
                "visibility": self.scale.expand(1, 1, 4),
                "heatmaps": self.scale.sigmoid().expand(1, 1, 4, 8, 8),
            }
            if kwargs.get("return_heatmap_logits"):
                output["heatmap_logits"] = self.scale.expand(1, 1, 4, 8, 8)
            return output

    model = FakeModel()
    transformed = transform_sample(_sample(), intervention="standard")
    criterion = HeatmapVLNLoss(heatmap_size=(8, 8), lambda_coord=0.0)

    loss, record = forward_loss(
        model,
        criterion,
        transformed,
        torch.device("cpu"),
        return_heatmap_logits=True,
    )

    assert torch.isfinite(loss)
    assert record["visibility"].shape == (1, 4, 4)
    assert record["heatmap_logits"].shape == (1, 4, 4, 8, 8)
    assert len(model.seen) == 4
    assert all(item["history_panoramas"].shape == (1, 1, 4, 3, 2, 2) for item in model.seen)
    assert all(item["current_views"].shape == (1, 4, 3, 2, 2) for item in model.seen)
    assert all(item["history_rel_poses"] is None for item in model.seen)
    assert all(item["return_heatmap_logits"] is True for item in model.seen)
    loss.backward()
    assert model.scale.grad is not None
    assert model.scale.grad.abs() > 0
    with pytest.raises(ValueError, match="non-None"):
        forward_loss(
            model,
            criterion,
            transformed,
            torch.device("cpu"),
            history_rel_poses=torch.zeros(4, 4),
        )


def test_blank_output_identity_gate_fails_closed_on_row_specific_results():
    visibility = torch.zeros(1, 4, 4)
    heatmaps = torch.full((1, 4, 4, 8, 8), 0.5)
    assert assert_blank_output_identity(visibility, heatmaps)["four_blank_chain_outputs_bitwise_identical"]

    visibility[:, 2] = 1
    with pytest.raises(RuntimeError, match="row-specific"):
        assert_blank_output_identity(visibility, heatmaps)


def test_blank_forward_runs_four_b1_calls_and_rejects_call_index_outputs():
    class RowSpecificModel:
        def __init__(self):
            self.calls = 0

        def __call__(self, **_kwargs):
            value = float(self.calls)
            self.calls += 1
            return {
                "visibility": torch.full((1, 1, 4), value, requires_grad=True),
                "heatmaps": torch.full((1, 1, 4, 8, 8), 0.25 + value * 0.1, requires_grad=True),
            }

    model = RowSpecificModel()
    transformed = transform_sample(_sample(), intervention="blank-images")
    criterion = HeatmapVLNLoss(heatmap_size=(8, 8), lambda_coord=0.0)

    with pytest.raises(RuntimeError, match="row-specific"):
        forward_loss(model, criterion, transformed, torch.device("cpu"))
    assert model.calls == 4


def test_history_permutation_gate_inverse_reorders_outputs_and_fails_on_drift():
    standard = {
        "sample_id": "sample-a",
        "visibility": torch.arange(16, dtype=torch.float32).reshape(1, 4, 4),
        "heatmaps": torch.arange(4 * 4 * 2 * 2, dtype=torch.float32).reshape(1, 4, 4, 2, 2),
    }
    standard["heatmap_logits"] = standard["heatmaps"] - 10.0
    permutation = [3, 2, 1, 0]
    shuffled = {
        "sample_id": "sample-a",
        "history_permutation": permutation,
        "visibility": standard["visibility"][:, permutation].clone(),
        "heatmaps": standard["heatmaps"][:, permutation].clone(),
        "heatmap_logits": standard["heatmap_logits"][:, permutation].clone(),
    }

    gate = assert_history_permutation_equivariance([standard], [shuffled])
    assert gate["passed"] is True
    assert gate["bitwise_exact"] is True

    shuffled["heatmaps"][0, 0, 0, 0, 0] += 1
    with pytest.raises(RuntimeError, match="equivariance failed"):
        assert_history_permutation_equivariance([standard], [shuffled])


def test_metrics_report_perfect_anchor_identity_for_four_distinct_targets():
    sample = _sample()
    prediction = sample["heatmap"].clone()
    visibility_logits = torch.where(sample["gt_visibility"].bool(), 10.0, -10.0)
    record = {
        "visibility": visibility_logits.unsqueeze(0),
        "heatmaps": prediction.unsqueeze(0),
        "gt_visibility": sample["gt_visibility"],
        "gt_heatmaps": sample["heatmap"],
    }

    aggregate = compute_metrics([record])

    assert aggregate["joint_pck8"] == 1.0
    assert aggregate["visible_view_accuracy"] == 1.0
    assert aggregate["anchor_identity"]["accuracy"] == 1.0
    assert aggregate["anchor_identity"]["chance"] == 0.25
    assert aggregate["anchor_identity"]["confusion_matrix"] == [
        [1, 0, 0, 0],
        [0, 1, 0, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 1],
    ]
    for slot in range(4):
        assert compute_metrics([record], slot=slot)["anchor_identity"]["accuracy"] == 1.0


def test_single_swap_reports_targeted_metrics_and_untargeted_output_invariance():
    sample = _sample()
    base = {
        "sample_id": "sample-a",
        "target_slot": None,
        "visibility": torch.where(sample["gt_visibility"].bool(), 10.0, -10.0).unsqueeze(0),
        "heatmaps": sample["heatmap"].unsqueeze(0),
        "gt_visibility": sample["gt_visibility"],
        "gt_heatmaps": sample["heatmap"],
    }
    swaps = []
    for target_slot in range(4):
        swapped = {key: value.clone() if torch.is_tensor(value) else value for key, value in base.items()}
        swapped["target_slot"] = target_slot
        swapped["heatmaps"][0, target_slot] = torch.roll(swapped["heatmaps"][0, target_slot], shifts=2, dims=-1)
        swaps.append(swapped)

    targeted = compute_metrics(swaps, dynamic_slots="targeted")
    untargeted = compute_metrics(swaps, dynamic_slots="untargeted")
    paired = paired_single_swap_output_change([base], swaps)

    assert targeted["visible_history_count"] == 4
    assert untargeted["visible_history_count"] == 12
    assert paired["targeted"]["comparisons"] == 4
    assert paired["untargeted"]["comparisons"] == 12
    assert paired["targeted"]["mean_heatmap_l1"] > 0
    assert paired["untargeted"]["mean_heatmap_l1"] == 0


def test_tensor_state_hash_covers_dtype_shape_and_exact_bfloat16_bytes():
    state = {
        "scalar": torch.tensor(1.0),
        "x": torch.tensor([1.0, 2.0], dtype=torch.bfloat16),
    }
    same = {name: value.clone() for name, value in state.items()}
    changed = {
        "scalar": torch.tensor(1.0),
        "x": torch.tensor([1.0, 3.0], dtype=torch.bfloat16),
    }

    assert tensor_state_sha256(state) == tensor_state_sha256(same)
    assert tensor_state_sha256(state) != tensor_state_sha256(changed)


class _TinyPilotModel(torch.nn.Module):
    def __init__(self, baseline_lora: torch.Tensor):
        super().__init__()
        self.lora_A = torch.nn.Parameter(baseline_lora.clone())
        self.heatmap_vln = torch.nn.Module()
        self.heatmap_vln.pose_free_matcher = torch.nn.Linear(2, 1, bias=False)


def _factorial_contracts(baseline_lora: torch.Tensor):
    baseline_lora_hash = tensor_state_sha256({"lora_A": baseline_lora})
    stage1 = {
        "path": "/pinned/stage1_s2.pth",
        "file_sha256": "stage1-file-hash",
        "matched_lora_tensors": 1,
        "loaded_lora_sha256": baseline_lora_hash,
    }
    manifest = {
        "path": "/pinned/manifest.json",
        "file_sha256": "manifest-file-hash",
        "manifest_sha256": "manifest-strong-hash",
        "source_inventory_sha256": "inventory-hash",
        "max_clip_id": 2000,
        "source_inventory_clips": 2000,
        "num_history": 4,
        "train_identity_sha256": "train-identity",
        "val_identity_sha256": "val-identity",
        "train_samples": 128,
        "val_samples": 40,
        "scene_disjoint": True,
        "split_source_inventories": {
            "train": {"inventory_sha256": "inventory-hash", "clips": 2000},
            "val": {"inventory_sha256": "inventory-hash", "clips": 2000},
        },
    }
    config = {
        "decoder_mode": "pose_free_matcher",
        "trajectory_enabled": False,
        "vit_layer_indices": [],
        "llm_layer_indices": [20],
        "model_pose_input": None,
        "isolated_pair_chains": True,
        "histories_per_qwen_chain": 1,
        "qwen_forward_batch_size": 1,
        "qwen_forwards_per_sample": 4,
    }
    runtime = {
        "decoder_mode": "pose_free_matcher",
        "trajectory_enabled": False,
        "vit_hooks": [],
        "llm_hooks": [20],
        "matcher_uses_relative_pose": False,
        "head_trainable_parameters": 2,
        "isolated_pair_chains": True,
        "histories_per_qwen_chain": 1,
        "history_anchor_number_per_chain": 1,
        "qwen_forward_batch_size": 1,
        "qwen_forwards_per_sample": 4,
    }
    return stage1, manifest, config, runtime


def _factorial_payload(
    *,
    branch: str,
    head: torch.Tensor,
    lora: torch.Tensor,
    baseline_lora: torch.Tensor,
    stage1: dict,
    manifest: dict,
    config: dict,
    runtime: dict,
) -> dict:
    head_state = {"weight": head.clone()}
    lora_state = {"lora_A": lora.clone()}
    return {
        "schema": CHECKPOINT_SCHEMA,
        "branch": branch,
        "step": 512,
        "head_state_dict": head_state,
        "lora_state_dict": lora_state,
        "head_state_sha256": tensor_state_sha256(head_state),
        "lora_state_sha256": tensor_state_sha256(lora_state),
        "initial_head_sha256": "shared-initial-head",
        "initial_lora_sha256": tensor_state_sha256({"lora_A": baseline_lora}),
        "expected_lora_tensors": 1,
        "stage1_s2_contract": stage1,
        "manifest_contract": manifest,
        "pose_free_config_contract": config,
        "runtime_contract": runtime,
        "training_sample_schedule_sha256": "shared-schedule",
    }


def test_factorial_eval_loads_only_head_from_strict_paired_head_only_checkpoint(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(pilot_module, "EXPECTED_LORA_TENSORS", 1)
    baseline_lora = torch.tensor([1.0])
    trained_lora = torch.tensor([9.0])
    joint_head = torch.tensor([[3.0, 4.0]])
    head_only_head = torch.tensor([[-2.0, 7.0]])
    stage1, manifest, config, runtime = _factorial_contracts(baseline_lora)
    joint_payload = _factorial_payload(
        branch="heatmap-lora",
        head=joint_head,
        lora=trained_lora,
        baseline_lora=baseline_lora,
        stage1=stage1,
        manifest=manifest,
        config=config,
        runtime=runtime,
    )
    head_payload = _factorial_payload(
        branch="head-only",
        head=head_only_head,
        lora=baseline_lora,
        baseline_lora=baseline_lora,
        stage1=stage1,
        manifest=manifest,
        config=config,
        runtime=runtime,
    )
    joint_path = tmp_path / "joint.pth"
    head_path = tmp_path / "head-only.pth"
    torch.save(joint_payload, joint_path)
    torch.save(head_payload, head_path)
    model = _TinyPilotModel(baseline_lora)

    _payload, contract = load_pilot_checkpoint_strict(
        model,
        str(joint_path),
        branch="heatmap-lora",
        stage1_contract=stage1,
        manifest_contract=manifest,
        eval_lora="trained",
        eval_head_checkpoint=str(head_path),
        runtime_contract=runtime,
        config_contract=config,
    )

    torch.testing.assert_close(model.lora_A, trained_lora)
    torch.testing.assert_close(model.heatmap_vln.pose_free_matcher.weight, head_only_head)
    assert contract["head_override"] is True
    assert contract["lora_source_checkpoint"] == {
        "path": str(joint_path.resolve()),
        "file_sha256": file_sha256(joint_path),
        "branch": "heatmap-lora",
        "head_state_sha256": joint_payload["head_state_sha256"],
        "lora_state_sha256": joint_payload["lora_state_sha256"],
    }
    assert contract["head_source_checkpoint"] == {
        "path": str(head_path.resolve()),
        "file_sha256": file_sha256(head_path),
        "branch": "head-only",
        "head_state_sha256": head_payload["head_state_sha256"],
        "lora_state_sha256": head_payload["lora_state_sha256"],
    }


def test_default_eval_still_loads_head_and_lora_from_the_same_pilot_checkpoint(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(pilot_module, "EXPECTED_LORA_TENSORS", 1)
    baseline_lora = torch.tensor([1.0])
    trained_lora = torch.tensor([9.0])
    joint_head = torch.tensor([[3.0, 4.0]])
    stage1, manifest, config, runtime = _factorial_contracts(baseline_lora)
    joint_payload = _factorial_payload(
        branch="heatmap-lora",
        head=joint_head,
        lora=trained_lora,
        baseline_lora=baseline_lora,
        stage1=stage1,
        manifest=manifest,
        config=config,
        runtime=runtime,
    )
    joint_path = tmp_path / "joint.pth"
    torch.save(joint_payload, joint_path)
    model = _TinyPilotModel(baseline_lora)

    _payload, contract = load_pilot_checkpoint_strict(
        model,
        str(joint_path),
        branch="heatmap-lora",
        stage1_contract=stage1,
        manifest_contract=manifest,
        eval_lora="trained",
        runtime_contract=runtime,
        config_contract=config,
    )

    torch.testing.assert_close(model.lora_A, trained_lora)
    torch.testing.assert_close(model.heatmap_vln.pose_free_matcher.weight, joint_head)
    assert contract["head_override"] is False
    assert contract["head_source_checkpoint"]["path"] == str(joint_path.resolve())
    assert contract["lora_source_checkpoint"]["path"] == str(joint_path.resolve())


@pytest.mark.parametrize(
    ("mutation", "error"),
    [
        ("head_hash", "head strong hash mismatch"),
        ("stage1", "Stage1-S2 contract mismatch"),
        ("manifest", "manifest contract mismatch"),
        ("runtime", "runtime contract differs"),
        ("schedule", "not paired"),
        ("frozen_lora", "supposedly frozen LoRA"),
    ],
)
def test_factorial_eval_rejects_unverified_head_checkpoint_mix(
    tmp_path,
    monkeypatch,
    mutation,
    error,
):
    monkeypatch.setattr(pilot_module, "EXPECTED_LORA_TENSORS", 1)
    baseline_lora = torch.tensor([1.0])
    stage1, manifest, config, runtime = _factorial_contracts(baseline_lora)
    joint_payload = _factorial_payload(
        branch="heatmap-lora",
        head=torch.tensor([[3.0, 4.0]]),
        lora=torch.tensor([9.0]),
        baseline_lora=baseline_lora,
        stage1=stage1,
        manifest=manifest,
        config=config,
        runtime=runtime,
    )
    head_payload = _factorial_payload(
        branch="head-only",
        head=torch.tensor([[-2.0, 7.0]]),
        lora=baseline_lora,
        baseline_lora=baseline_lora,
        stage1=stage1,
        manifest=manifest,
        config=config,
        runtime=runtime,
    )
    if mutation == "head_hash":
        head_payload["head_state_sha256"] = "corrupt"
    elif mutation == "stage1":
        head_payload["stage1_s2_contract"] = {**stage1, "file_sha256": "other"}
    elif mutation == "manifest":
        head_payload["manifest_contract"] = {**manifest, "manifest_sha256": "other"}
    elif mutation == "runtime":
        head_payload["runtime_contract"] = {**runtime, "head_trainable_parameters": 999}
    elif mutation == "schedule":
        head_payload["training_sample_schedule_sha256"] = "other"
    elif mutation == "frozen_lora":
        changed = {"lora_A": torch.tensor([2.0])}
        head_payload["lora_state_dict"] = changed
        head_payload["lora_state_sha256"] = tensor_state_sha256(changed)
    joint_path = tmp_path / "joint.pth"
    head_path = tmp_path / "head-only.pth"
    torch.save(joint_payload, joint_path)
    torch.save(head_payload, head_path)

    with pytest.raises(RuntimeError, match=error):
        load_pilot_checkpoint_strict(
            _TinyPilotModel(baseline_lora),
            str(joint_path),
            branch="heatmap-lora",
            stage1_contract=stage1,
            manifest_contract=manifest,
            eval_lora="trained",
            eval_head_checkpoint=str(head_path),
            runtime_contract=runtime,
            config_contract=config,
        )

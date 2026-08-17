from __future__ import annotations

from types import SimpleNamespace

import pytest
import scripts.tools.train_pose_free_visual_identity_pilot as pilot
import torch
import yaml


def _args(**overrides):
    values = {
        "train_mode": "head-warmup",
        "warmup_checkpoint": None,
        "train_steps": 1,
        "grad_clip": 1.0,
        "log_every": 1,
        "head_learning_rate": 1e-4,
        "lora_learning_rate": 5e-5,
        "weight_decay": 1e-2,
        "seed": 42,
        "max_trainable_lora_layer": 20,
        "device": "cpu",
        "data_root": "/data",
        "model_path": None,
    }
    values.update(overrides)
    return SimpleNamespace(**values)


def _transformed_sample(size: int = 16) -> dict:
    histories = torch.randn(4, 4, 3, 2, 2)
    gt_heatmaps = torch.zeros(4, 4, size, size)
    gt_visibility = torch.zeros(4, 4)
    for slot, (view, y, x) in enumerate(((0, 2, 2), (1, 2, 2), (2, 2, 2), (3, 2, 2))):
        gt_visibility[slot, view] = 1
        gt_heatmaps[slot, view, y, x] = 1
    return {
        "current_views": torch.randn(4, 3, 2, 2),
        "current_frame": torch.randn(3, 2, 2),
        "history_panoramas": histories,
        "history_frames": histories[:, 0].clone(),
        "gt_visibility": gt_visibility,
        "gt_heatmaps": gt_heatmaps,
        "sample_id": "sample-a",
        "metadata": {"intervention": "standard", "target_slot": None},
    }


def test_state_machine_requires_one_shared_warmup_checkpoint():
    pilot.validate_args(_args())
    with pytest.raises(ValueError, match="forbids --warmup-checkpoint"):
        pilot.validate_args(_args(warmup_checkpoint="warmup.pth"))
    for mode in ("lora-identity", "lora-heatmap-control"):
        with pytest.raises(ValueError, match="requires --warmup-checkpoint"):
            pilot.validate_args(_args(train_mode=mode))
        pilot.validate_args(_args(train_mode=mode, warmup_checkpoint="warmup.pth"))
    with pytest.raises(ValueError, match="layer at 20"):
        pilot.validate_args(_args(max_trainable_lora_layer=19))


def test_cli_default_lora_learning_rate_is_five_e_minus_five():
    args = pilot.parse_args(
        [
            "--train-mode",
            "head-warmup",
            "--config",
            "config.yaml",
            "--checkpoint",
            "stage1.pth",
            "--selection-manifest",
            "manifest.json",
            "--data-root",
            "/data",
            "--output-dir",
            "/output",
        ]
    )
    assert args.lora_learning_rate == 5e-5


def test_config_forces_visual_equal_view_query_and_strict_b1(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "data": {
                    "root": "old",
                    "image_size": [224, 224],
                    "init_hm_size": [64, 64],
                },
                "model": {
                    "device": "cpu",
                    "llm": {"model_path": "/model", "lora_dropout": 0.5},
                    "heatmap": {
                        "decoder_mode": "legacy",
                        "trajectory": {"enable": True},
                        "vit_layer_indices": [7, 15],
                        "llm_layer_indices": [6, 13, 20],
                        "pose_free": {"history_query_source": "text_anchor"},
                    },
                    "action_head": {"enable": True},
                },
                "loss": {"heatmap_vln": {}},
            }
        ),
        encoding="utf-8",
    )
    args = _args(
        config=str(config_path),
        train_mode="lora-identity",
        warmup_checkpoint="warmup.pth",
    )

    cfg = pilot.load_visual_identity_config(args)
    contract = pilot.visual_identity_config_contract(cfg)

    assert cfg["model"]["heatmap"]["pose_free"]["history_query_source"] == pilot.HISTORY_QUERY_SOURCE
    assert cfg["model"]["llm"]["gradient_checkpointing"] is True
    assert contract["history_query_source"] == pilot.HISTORY_QUERY_SOURCE
    assert contract["history_visual_views_per_query"] == 4
    assert contract["history_visual_view_reduction"] == "equal_weight_mean"
    assert contract["qwen_forward_batch_size"] == 1
    assert contract["qwen_forwards_per_sample"] == 4
    assert contract["raw_heatmap_logits_required"] is True


def test_registered_loss_contrast_is_exact():
    base = torch.tensor(3.0, requires_grad=True)
    identity = torch.tensor(5.0, requires_grad=True)
    panorama = torch.tensor(7.0, requires_grad=True)
    identity_output = {
        "identity_loss": identity,
        "panorama_loss": panorama,
        "total": 2 * identity + panorama,
    }

    total, components = pilot.compose_training_loss(
        "lora-identity",
        base,
        identity_output,
    )
    assert total.item() == 20.0
    assert components == {
        "base": base,
        "identity": identity,
        "panorama": panorama,
        "total": total,
    }
    total.backward()
    assert base.grad.item() == 1.0
    assert identity.grad.item() == 2.0
    assert panorama.grad.item() == 1.0

    control, control_components = pilot.compose_training_loss(
        "lora-heatmap-control",
        torch.tensor(3.0),
        {
            "identity_loss": torch.tensor(5.0),
            "panorama_loss": torch.tensor(7.0),
            "total": torch.tensor(17.0),
        },
    )
    assert control.item() == 10.0
    assert control_components["identity"].item() == 0.0
    assert control_components["panorama"].item() == 7.0
    with pytest.raises(RuntimeError, match="must not consume"):
        pilot.compose_training_loss("head-warmup", base, identity_output)


def test_raw_logit_regroup_preserves_graph():
    source = torch.randn(4, 1, 4, 8, 8, requires_grad=True)
    outputs = []
    for slot in range(4):
        logits = source[slot : slot + 1]
        outputs.append(
            {
                "visibility": logits.mean(dim=(-1, -2)),
                "heatmaps": logits.sigmoid(),
                "heatmap_logits": logits,
            }
        )

    visibility, heatmaps, logits = pilot.regroup_visual_identity_outputs(
        outputs,
        num_histories=4,
    )

    assert visibility.shape == (1, 4, 4)
    assert heatmaps.shape == (1, 4, 4, 8, 8)
    assert logits.shape == (1, 4, 4, 8, 8)
    assert logits.grad_fn is not None
    logits.sum().backward()
    torch.testing.assert_close(source.grad, torch.ones_like(source))


def test_auxiliary_autograd_audit_retains_graph_for_total_backward():
    lora = torch.nn.Parameter(torch.tensor([2.0]))
    auxiliary = (lora.square()).sum()
    base = (3.0 * lora).sum()
    total = base + auxiliary

    audit = pilot.audit_identity_auxiliary_gradient(
        auxiliary,
        {"model.layers.0.adapter.lora_A": lora},
    )

    assert audit["tensors_with_nonzero_grad"] == 1
    assert audit["layers_with_nonzero_grad"] == [0]
    assert audit["total_grad_norm"] == 4.0
    assert lora.grad is None
    total.backward()
    torch.testing.assert_close(lora.grad, torch.tensor([7.0]))


def test_component_gradient_audit_can_report_zero_without_hiding_it():
    lora = torch.nn.Parameter(torch.tensor([2.0]))
    zero_panorama = (lora * 0.0).sum()
    parameters = {"model.layers.0.adapter.lora_A": lora}

    audit = pilot.audit_lora_objective_gradient(
        zero_panorama,
        parameters,
        objective_label="global_panorama_pixel_ce",
        require_nonzero=False,
    )

    assert audit["objective"] == "global_panorama_pixel_ce"
    assert audit["nonzero_gradient_reached"] is False
    assert audit["tensors_with_nonzero_grad"] == 0
    assert audit["total_grad_norm"] == 0
    with pytest.raises(RuntimeError, match="global_panorama_pixel_ce produced zero"):
        pilot.audit_lora_objective_gradient(
            zero_panorama,
            parameters,
            objective_label="global_panorama_pixel_ce",
            require_nonzero=True,
        )


def test_forward_executes_four_b1_raw_opt_in_calls_and_identity_reaches_source():
    class FakeModel:
        def __init__(self):
            self.scale = torch.nn.Parameter(torch.tensor(0.3))
            self.calls = []

        def __call__(self, **kwargs):
            self.calls.append(kwargs)
            logits = self.scale.expand(1, 1, 4, 16, 16)
            return {
                "visibility": self.scale.expand(1, 1, 4),
                "heatmaps": logits.sigmoid(),
                "heatmap_logits": logits,
            }

    class FakeBaseLoss:
        def __call__(self, _vis, heatmaps, **_kwargs):
            return {"total": heatmaps.mean()}

    class FakeIdentityLoss:
        def __call__(self, logits, *_args):
            identity = logits.mean()
            panorama = logits.square().mean()
            return {
                "identity_loss": identity,
                "panorama_loss": panorama,
                "view_loss": panorama,
                "within_view_loss": panorama,
                "total": 2 * identity + panorama,
                "minimum_target_separation": logits.new_tensor(12.0),
            }

    model = FakeModel()
    loss, record = pilot.forward_visual_identity_loss(
        model,
        FakeBaseLoss(),
        FakeIdentityLoss(),
        _transformed_sample(),
        torch.device("cpu"),
        train_mode="lora-identity",
    )

    assert len(model.calls) == 4
    assert all(call["video_frames"].shape[0] == 1 for call in model.calls)
    assert all(call["history_panoramas"].shape[:2] == (1, 1) for call in model.calls)
    assert all(call["return_heatmap_logits"] is True for call in model.calls)
    assert all(call["history_rel_poses"] is None for call in model.calls)
    assert record["heatmap_logits"].shape == (1, 4, 4, 16, 16)
    assert record["current_prefix_identity_gate"]["passed"] is True
    assert record["current_patch_identity_gate"] is None
    assert record["_base_term_graph"].requires_grad
    assert record["_identity_term_graph"].requires_grad
    assert record["_panorama_term_graph"].requires_grad
    loss.backward()
    assert model.scale.grad is not None
    assert model.scale.grad.abs() > 0


class _Adapter(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.lora_A = torch.nn.Parameter(torch.tensor([1.0]))


class _TinyStateModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qwen2_5_vl = torch.nn.Module()
        self.qwen2_5_vl.model = torch.nn.Module()
        self.qwen2_5_vl.model.layers = torch.nn.ModuleList([_Adapter()])
        self.heatmap_vln = torch.nn.Module()
        self.heatmap_vln.pose_free_matcher = torch.nn.Linear(2, 1, bias=False)


def test_trainable_state_is_head_xor_reachable_lora(monkeypatch):
    monkeypatch.setattr(pilot, "EXPECTED_LORA_TENSORS", 1)
    monkeypatch.setattr(pilot, "EXPECTED_TRAINABLE_LORA_TENSORS", 1)
    monkeypatch.setattr(pilot, "EXPECTED_TRAINABLE_LORA_LAYERS", (0,))
    model = _TinyStateModel()

    head, lora = pilot.configure_training_state(model, "head-warmup")
    assert head and not lora
    assert all(parameter.requires_grad for parameter in head.values())
    assert model.qwen2_5_vl.model.layers[0].lora_A.requires_grad is False

    head, lora = pilot.configure_training_state(model, "lora-identity")
    assert not head and lora
    assert model.heatmap_vln.pose_free_matcher.weight.requires_grad is False
    assert model.qwen2_5_vl.model.layers[0].lora_A.requires_grad is True
    contract = pilot.build_optimization_contract(
        _args(train_mode="lora-identity", warmup_checkpoint="warmup.pth"),
        {"model": {"llm": {"gradient_checkpointing": True}}},
        head,
        lora,
    )
    assert contract["protocol_reachable_lora_tensors"] == 1
    assert contract["actual_trainable_lora_tensors"] == 1
    assert contract["actual_trainable_lora_layers"] == [0]
    assert contract["learning_rates"]["active"] == 5e-5


def _contracts(lora_hash: str):
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
        "history_query_source": pilot.HISTORY_QUERY_SOURCE,
        "qwen_forward_batch_size": 1,
        "qwen_forwards_per_sample": 4,
    }
    runtime = {
        "history_query_source": pilot.HISTORY_QUERY_SOURCE,
        "qwen_forward_batch_size": 1,
        "qwen_forwards_per_sample": 4,
    }
    return stage1, manifest, config, runtime


def _optimization_contract(train_mode: str):
    lora_mode = train_mode != "head-warmup"
    return {
        "optimizer": {
            "name": "AdamW",
            "betas": [0.9, 0.999],
            "eps": 1e-8,
            "weight_decay": 0.01,
            "amsgrad": False,
        },
        "train_steps": 1,
        "seed": 42,
        "learning_rates": {
            "head": 1e-4,
            "lora": 5e-5,
            "active_group": "reachable_lora" if lora_mode else "pose_free_matcher_warmup",
            "active": 5e-5 if lora_mode else 1e-4,
        },
        "grad_clip": 1.0,
        "max_trainable_lora_layer": 20,
        "gradient_checkpointing": lora_mode,
        "protocol_reachable_lora_tensors": 168,
        "protocol_reachable_lora_layers": list(range(21)),
        "expected_trainable_lora_tensors": 168 if lora_mode else 0,
        "actual_trainable_lora_tensors": 168 if lora_mode else 0,
        "expected_trainable_lora_layers": list(range(21)) if lora_mode else [],
        "actual_trainable_lora_layers": list(range(21)) if lora_mode else [],
        "actual_trainable_head_tensors": 0 if lora_mode else 7,
        "qwen_forward_batch_size": 1,
        "qwen_forwards_per_sample": 4,
    }


def _warmup_payload(model, stage1, manifest, config, runtime):
    head = pilot.base_pilot.pose_free_head_state_dict(model)
    lora = pilot.base_pilot.lora_state_dict(model)
    head_hash = pilot.base_pilot.tensor_state_sha256(head)
    lora_hash = pilot.base_pilot.tensor_state_sha256(lora)
    return {
        "schema": pilot.CHECKPOINT_SCHEMA,
        "protocol": pilot.PROTOCOL,
        "train_mode": "head-warmup",
        "step": 1,
        "head_state_dict": head,
        "lora_state_dict": lora,
        "head_state_sha256": head_hash,
        "lora_state_sha256": lora_hash,
        "initial_head_sha256": "initial-head",
        "initial_lora_sha256": lora_hash,
        "expected_lora_tensors": 1,
        "stage1_s2_contract": stage1,
        "manifest_contract": manifest,
        "pose_free_config_contract": config,
        "runtime_contract": runtime,
        "warmup_checkpoint_contract": None,
        "training_sample_schedule_sha256": "schedule",
        "optimization_contract": _optimization_contract("head-warmup"),
        "loss_contract": pilot.expected_loss_contract("head-warmup"),
    }


def test_v3_warmup_load_is_exact_and_legacy_joint_checkpoint_is_rejected(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(pilot, "EXPECTED_LORA_TENSORS", 1)
    source_model = _TinyStateModel()
    with torch.no_grad():
        source_model.heatmap_vln.pose_free_matcher.weight.copy_(torch.tensor([[4.0, -3.0]]))
    lora_hash = pilot.base_pilot.tensor_state_sha256(pilot.base_pilot.lora_state_dict(source_model))
    stage1, manifest, config, runtime = _contracts(lora_hash)
    payload = _warmup_payload(source_model, stage1, manifest, config, runtime)
    warmup_path = tmp_path / "warmup.pth"
    torch.save(payload, warmup_path)

    target_model = _TinyStateModel()
    initial_lora = target_model.qwen2_5_vl.model.layers[0].lora_A.detach().clone()
    contract = pilot.load_warmup_head_strict(
        target_model,
        warmup_path,
        stage1_contract=stage1,
        manifest_contract=manifest,
        config_contract=config,
        runtime_contract=runtime,
    )
    torch.testing.assert_close(
        target_model.heatmap_vln.pose_free_matcher.weight,
        source_model.heatmap_vln.pose_free_matcher.weight,
    )
    torch.testing.assert_close(target_model.qwen2_5_vl.model.layers[0].lora_A, initial_lora)
    assert contract["head_state_sha256"] == payload["head_state_sha256"]

    legacy_path = tmp_path / "legacy-joint.pth"
    torch.save(
        {
            "schema": pilot.base_pilot.CHECKPOINT_SCHEMA,
            "branch": "heatmap-lora",
            "runtime_contract": {"qwen_forward_batch_size": 4},
        },
        legacy_path,
    )
    with pytest.raises(RuntimeError, match="legacy anchor/B=4/joint"):
        pilot.validate_visual_identity_checkpoint_payload_strict(
            legacy_path,
            expected_train_mode="head-warmup",
            stage1_contract=stage1,
            manifest_contract=manifest,
            config_contract=config,
            runtime_contract=runtime,
        )


@pytest.mark.parametrize("train_mode", ["lora-identity", "lora-heatmap-control"])
def test_lora_checkpoint_contract_requires_same_frozen_warmup_head(
    tmp_path,
    monkeypatch,
    train_mode,
):
    monkeypatch.setattr(pilot, "EXPECTED_LORA_TENSORS", 1)
    model = _TinyStateModel()
    lora_hash = pilot.base_pilot.tensor_state_sha256(pilot.base_pilot.lora_state_dict(model))
    stage1, manifest, config, runtime = _contracts(lora_hash)
    warmup_payload = _warmup_payload(model, stage1, manifest, config, runtime)
    warmup_contract = {
        "schema": pilot.CHECKPOINT_SCHEMA,
        "protocol": pilot.PROTOCOL,
        "path": "/warmup.pth",
        "file_sha256": "warmup-file",
        "head_state_sha256": warmup_payload["head_state_sha256"],
        "lora_state_sha256": warmup_payload["lora_state_sha256"],
        "step": 1,
        "training_sample_schedule_sha256": "schedule",
        "optimization_contract": _optimization_contract("head-warmup"),
    }
    payload = {
        **warmup_payload,
        "train_mode": train_mode,
        "initial_head_sha256": warmup_contract["head_state_sha256"],
        "warmup_checkpoint_contract": warmup_contract,
        "optimization_contract": _optimization_contract(train_mode),
        "loss_contract": pilot.expected_loss_contract(train_mode),
    }
    path = tmp_path / f"{train_mode}.pth"
    torch.save(payload, path)
    pilot.validate_visual_identity_checkpoint_payload_strict(
        path,
        expected_train_mode=train_mode,
        stage1_contract=stage1,
        manifest_contract=manifest,
        config_contract=config,
        runtime_contract=runtime,
    )

    payload["head_state_dict"] = {
        "weight": payload["head_state_dict"]["weight"] + 1,
    }
    payload["head_state_sha256"] = pilot.base_pilot.tensor_state_sha256(payload["head_state_dict"])
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="bitwise-frozen warmup head"):
        pilot.validate_visual_identity_checkpoint_payload_strict(
            path,
            expected_train_mode=train_mode,
            stage1_contract=stage1,
            manifest_contract=manifest,
            config_contract=config,
            runtime_contract=runtime,
        )


def test_identity_control_pair_requires_exact_provenance_schedule_and_optimization(
    tmp_path,
    monkeypatch,
):
    monkeypatch.setattr(pilot, "EXPECTED_LORA_TENSORS", 1)
    model = _TinyStateModel()
    lora_hash = pilot.base_pilot.tensor_state_sha256(pilot.base_pilot.lora_state_dict(model))
    stage1, manifest, config, runtime = _contracts(lora_hash)
    warmup_payload = _warmup_payload(model, stage1, manifest, config, runtime)
    warmup_contract = {
        "schema": pilot.CHECKPOINT_SCHEMA,
        "protocol": pilot.PROTOCOL,
        "path": "/same/warmup.pth",
        "file_sha256": "actual-warmup-file-sha",
        "head_state_sha256": warmup_payload["head_state_sha256"],
        "lora_state_sha256": warmup_payload["lora_state_sha256"],
        "step": 1,
        "training_sample_schedule_sha256": "warmup-schedule",
        "optimization_contract": _optimization_contract("head-warmup"),
    }

    def payload(mode):
        return {
            **warmup_payload,
            "train_mode": mode,
            "initial_head_sha256": warmup_contract["head_state_sha256"],
            "initial_lora_sha256": lora_hash,
            "warmup_checkpoint_contract": warmup_contract,
            "training_sample_schedule_sha256": "identical-lora-schedule",
            "optimization_contract": _optimization_contract(mode),
            "loss_contract": pilot.expected_loss_contract(mode),
        }

    identity_path = tmp_path / "identity.pth"
    control_path = tmp_path / "control.pth"
    torch.save(payload("lora-identity"), identity_path)
    torch.save(payload("lora-heatmap-control"), control_path)

    contract = pilot.validate_identity_control_checkpoint_pair(identity_path, control_path)
    assert contract["passed"] is True
    assert contract["only_registered_difference"] == {
        "identity_weight": [2.0, 0.0],
    }
    assert "optimization_contract" in contract["matched_contracts"]
    assert "warmup_actual_file_sha256" in contract["matched_contracts"]

    changed = payload("lora-heatmap-control")
    changed["optimization_contract"]["learning_rates"]["lora"] = 6e-5
    changed["optimization_contract"]["learning_rates"]["active"] = 6e-5
    torch.save(changed, control_path)
    with pytest.raises(RuntimeError, match="not a causal pair: optimization_contract"):
        pilot.validate_identity_control_checkpoint_pair(identity_path, control_path)


def test_strict_optimization_contract_rejects_wrong_reachable_layer_set():
    contract = _optimization_contract("lora-identity")
    contract["actual_trainable_lora_layers"] = list(range(20))
    with pytest.raises(RuntimeError, match="actual LoRA layers"):
        pilot.validate_optimization_contract_strict(
            contract,
            expected_train_mode="lora-identity",
            expected_step=1,
        )

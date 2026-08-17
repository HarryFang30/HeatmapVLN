"""Security boundary tests for heatmap-control checkpoint resume."""

from __future__ import annotations

import ast
import copy
import hashlib
from pathlib import Path

import pytest
import torch
import torch.nn as nn

from scripts.training.checkpoint import load_checkpoint_for_resume
from scripts.training.heatmap_control_resume import (
    HeatmapControlResumeError,
    reject_heatmap_control_load_weights,
    validate_heatmap_control_resume_checkpoint,
)
from scripts.training.native_internnav_dependency import (
    NATIVE_DEPENDENCY_SCHEMA,
    NATIVE_MODEL_FILE_COUNT,
    NATIVE_MODEL_MANIFEST_PATH,
    NATIVE_MODEL_MANIFEST_SHA256,
    NATIVE_MODEL_PATH,
)


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.heatmap_control = nn.Linear(3, 3)


class _ControlModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.native_qwen = nn.Linear(3, 3)
        self.heatmap_tokenizer = nn.Linear(3, 2)
        self.nextdit_action_head = nn.Module()
        self.nextdit_action_head.traj_dit = nn.Module()
        self.nextdit_action_head.traj_dit.model = nn.Module()
        self.nextdit_action_head.traj_dit.model.layers = nn.ModuleList(
            [_Block(), _Block()]
        )


def _control_state(model: nn.Module) -> dict[str, torch.Tensor]:
    return {
        name: parameter.detach().clone()
        for name, parameter in model.named_parameters()
        if name.startswith("heatmap_tokenizer.")
        or (
            name.startswith("nextdit_action_head.traj_dit.model.layers.")
            and ".heatmap_control." in name
        )
    }


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    digest.update(path.read_bytes())
    return digest.hexdigest()


def _base_config(tmp_path: Path) -> dict:
    dependency_path = tmp_path / "frozen_heatmap.pth"
    torch.save(
        {"trainable_state_dict": {"heatmap_vln.decoder.weight": torch.ones(2, 2)}},
        dependency_path,
    )
    dependency_sha = _sha256(dependency_path)
    return {
        "model": {
            "llm": {
                "model_path": "/models/original-internnav",
                "use_lora": False,
            },
            "action_head": {
                "enable": True,
                "nextdit": {
                    "enabled": True,
                    "internnav_model_path": "/models/original-internnav",
                    "internnav_system1_path": "",
                    "pretrained_system1_path": None,
                    "dav2_ckpt_path": "",
                    "warmup_steps": 0,
                    "pano_latent_adapter": {
                        "enabled": False,
                        "pretrained_path": "",
                    },
                    "heatmap_control": {
                        "enabled": True,
                        "schema_version": "heatmap-control-v1",
                        "token_dim": 128,
                        "control_dim": 128,
                        "num_heads": 4,
                        "coarse_size": 8,
                        "temporal_layers": 1,
                        "temporal_heads": 4,
                        "temporal_ffn_dim": 512,
                        "dropout": 0.0,
                        "age_normalizer_steps": 32.0,
                        "heatmap_checkpoint_path": str(dependency_path),
                        "heatmap_checkpoint_sha256": dependency_sha,
                    },
                },
            },
        },
        "training": {
            "stages": [
                {
                    "name": "heatmap_control",
                    "trainable_modules": [
                        "heatmap_tokenizer",
                        "heatmap_control",
                    ],
                    "strict_trainable_modules": True,
                    "train_action": True,
                    "train_heatmap": False,
                    "train_history": False,
                    "train_future": False,
                    "train_lm": False,
                    "train_system2_sft": False,
                    "requires_base_checkpoint": False,
                    "bridge_only": False,
                }
            ]
        },
        "runtime": {},
    }


def _dependency_contract(cfg: dict) -> dict:
    control = cfg["model"]["action_head"]["nextdit"]["heatmap_control"]
    return {
        "schema_version": "frozen-heatmap-checkpoint-v1",
        "checkpoint_sha256": control["heatmap_checkpoint_sha256"],
        "target_module": "heatmap_vln",
        "frozen": True,
        "tensor_count": 1,
    }


def _payload(model: nn.Module, cfg: dict) -> dict:
    saved_cfg = copy.deepcopy(cfg)
    saved_cfg["runtime"]["frozen_heatmap_dependency"] = _dependency_contract(cfg)
    state = _control_state(model)
    payload = {
        "epoch": 3,
        "stage_idx": 0,
        "stage_name": "heatmap_control",
        "config": saved_cfg,
        "trainable_state_dict": copy.deepcopy(state),
        "online_trainable_state_dict": copy.deepcopy(state),
        "ema_state_dict": {
            "shadow": copy.deepcopy(state),
            "target_decay": 0.999,
            "warmup_steps": 10,
            "step_count": 20,
        },
        "optimizer_state_dict": {
            "state": {},
            "param_groups": [{"params": list(range(len(state)))}],
        },
        "scheduler_state_dict": {
            "last_epoch": 3,
            "_step_count": 4,
        },
    }
    mixture = cfg.get("data", {}).get("mixture")
    if mixture is not None:
        payload["mixture_sampler_state"] = {
            "schema": "heatmapvln-deterministic-mixture-sampler-v1",
            "epoch": 3,
            "seed": mixture["seed"],
            "requested_epoch_size": mixture["epoch_size"],
            "global_epoch_size": mixture["epoch_size"],
            "num_replicas": 8,
            "rank": 0,
            "drop_last": True,
            "profile": mixture["profile"],
            "weights": {
                "expert": 0.5,
                "dagger_normal": 0.2,
                "dagger_hard": 0.3,
            },
        }
    return payload


def _enable_exact_mixture_resume(cfg: dict) -> None:
    cfg["model"]["llm"]["model_path"] = NATIVE_MODEL_PATH
    cfg["model"]["action_head"]["nextdit"][
        "internnav_model_path"
    ] = NATIVE_MODEL_PATH
    cfg["runtime"]["native_internnav_dependency"] = {
        "schema": NATIVE_DEPENDENCY_SCHEMA,
        "model_path": NATIVE_MODEL_PATH,
        "manifest_path": NATIVE_MODEL_MANIFEST_PATH,
        "manifest_sha256": NATIVE_MODEL_MANIFEST_SHA256,
        "file_count": NATIVE_MODEL_FILE_COUNT,
        "verified": True,
    }
    cfg["data"] = {
        "root": "/datasets/expert",
        "train_split": "train",
        "image_size": [384, 384],
        "init_hm_size": [64, 64],
        "dataset_type": "expert_dagger_mixture",
        "in_order": True,
        "trajectory": {
            "sample_stride": 2,
            "predict_horizon": 32,
            "trajectory_target_convention": "internnav_habitat",
        },
        "trajectory_dagger": {
            "collection_roots": [
                f"/datasets/dagger/shard_{index:02d}" for index in range(4)
            ],
            "source_types": ["dagger_normal", "dagger_hard"],
            "num_history": 8,
            "expected_policy_mode": "internnav_native",
            "expected_policy_fingerprint": "internnav-native-v1:" + "b" * 64,
        },
        "mixture": {
            "profile": "expert50_normal20_hard30",
            "weights": None,
            "epoch_size": 72000,
            "seed": 42,
        },
    }
    cfg["optim"] = {
        "optimizer": "adamw",
        "learning_rate": 5.0e-5,
        "heatmap_tokenizer_lr": 1.0e-4,
        "heatmap_control_lr": 5.0e-5,
        "heatmap_gate_lr": 1.0e-4,
        "weight_decay": 0.01,
        "grad_clip": 1.0,
        "amp": "bf16",
        "scheduler": "cosine",
        "warmup_ratio": 0.03,
        "min_lr": 1.0e-6,
        "batch_size": 1,
        "grad_accum_steps": 4,
        "ema_decay": 0.999,
        "ema_warmup_steps": 500,
    }
    cfg["gpu"] = {"devices": list(range(8))}
    cfg["log"] = {"mid_epoch_save_every": 1000}
    cfg["seed"] = 42
    cfg["training"]["stages"][0]["epochs"] = 3


def _set_nested(config: dict, path: str, value: object) -> None:
    target: object = config
    parts = path.split(".")
    for part in parts[:-1]:
        target = target[int(part)] if isinstance(target, list) else target[part]
    if isinstance(target, list):
        target[int(parts[-1])] = value
    else:
        target[parts[-1]] = value


def _write_resume(tmp_path: Path, payload: dict, name: str = "resume.pth") -> Path:
    path = tmp_path / name
    torch.save(payload, path)
    return path


def test_valid_resume_checks_both_states_and_ema_without_mutation(tmp_path: Path) -> None:
    model = _ControlModel()
    cfg = _base_config(tmp_path)
    payload = _payload(model, cfg)
    path = _write_resume(tmp_path, payload)
    before = {name: value.clone() for name, value in model.state_dict().items()}

    report = validate_heatmap_control_resume_checkpoint(path, model, cfg)

    assert report["validated_trainable_state"] is True
    assert report["validated_online_state"] is True
    assert report["validated_ema_shadow"] is True
    assert report["validated_optimizer_state"] is True
    assert report["validated_scheduler_state"] is True
    assert report["validated_scaler_state"] is False
    assert report["state_tensor_count"] == len(_control_state(model))
    assert all(torch.equal(before[name], value) for name, value in model.state_dict().items())


def test_current_loaded_dependency_metadata_is_also_checked(tmp_path: Path) -> None:
    model = _ControlModel()
    cfg = _base_config(tmp_path)
    cfg["runtime"]["frozen_heatmap_dependency"] = _dependency_contract(cfg)
    path = _write_resume(tmp_path, _payload(model, cfg))

    validate_heatmap_control_resume_checkpoint(path, model, cfg)

    cfg["runtime"]["frozen_heatmap_dependency"]["tensor_count"] = 2
    with pytest.raises(HeatmapControlResumeError, match="current config runtime"):
        validate_heatmap_control_resume_checkpoint(path, model, cfg)


def test_module_prefixes_are_canonicalized(tmp_path: Path) -> None:
    model = _ControlModel()
    cfg = _base_config(tmp_path)
    payload = _payload(model, cfg)
    for state_name in (
        "trainable_state_dict",
        "online_trainable_state_dict",
    ):
        payload[state_name] = {
            "module." + name: value for name, value in payload[state_name].items()
        }
    payload["ema_state_dict"]["shadow"] = {
        name.replace(".heatmap_control.", ".module.heatmap_control."): value
        for name, value in payload["ema_state_dict"]["shadow"].items()
    }

    report = validate_heatmap_control_resume_checkpoint(
        _write_resume(tmp_path, payload), model, cfg
    )
    assert report["state_tensor_count"] == len(_control_state(model))


@pytest.mark.parametrize(
    "state_name",
    ["trainable_state_dict", "online_trainable_state_dict"],
)
def test_resume_state_rejects_native_tensor(tmp_path: Path, state_name: str) -> None:
    model = _ControlModel()
    cfg = _base_config(tmp_path)
    payload = _payload(model, cfg)
    payload[state_name]["native_qwen.weight"] = torch.ones(3, 3)

    with pytest.raises(HeatmapControlResumeError, match="unexpected=.*native_qwen"):
        validate_heatmap_control_resume_checkpoint(
            _write_resume(tmp_path, payload), model, cfg
        )


def test_ema_shadow_rejects_native_tensor(tmp_path: Path) -> None:
    model = _ControlModel()
    cfg = _base_config(tmp_path)
    payload = _payload(model, cfg)
    payload["ema_state_dict"]["shadow"]["native_qwen.bias"] = torch.ones(3)

    with pytest.raises(HeatmapControlResumeError, match="ema_state_dict.shadow.*unexpected"):
        validate_heatmap_control_resume_checkpoint(
            _write_resume(tmp_path, payload), model, cfg
        )


def test_resume_requires_exact_ema_shadow(tmp_path: Path) -> None:
    model = _ControlModel()
    cfg = _base_config(tmp_path)
    payload = _payload(model, cfg)
    payload.pop("ema_state_dict")

    with pytest.raises(HeatmapControlResumeError, match="ema_state_dict must be a mapping"):
        validate_heatmap_control_resume_checkpoint(
            _write_resume(tmp_path, payload), model, cfg
        )


@pytest.mark.parametrize(
    ("missing_key", "message"),
    [
        ("online_trainable_state_dict", "online_trainable_state_dict"),
        ("optimizer_state_dict", "optimizer_state_dict"),
        ("scheduler_state_dict", "scheduler_state_dict"),
    ],
)
def test_resume_requires_exact_online_optimizer_and_scheduler_state(
    tmp_path: Path,
    missing_key: str,
    message: str,
) -> None:
    model = _ControlModel()
    cfg = _base_config(tmp_path)
    payload = _payload(model, cfg)
    payload.pop(missing_key)

    with pytest.raises(HeatmapControlResumeError, match=message):
        validate_heatmap_control_resume_checkpoint(
            _write_resume(tmp_path, payload), model, cfg
        )


def test_fp16_resume_requires_scaler_but_bf16_does_not(tmp_path: Path) -> None:
    model = _ControlModel()
    cfg = _base_config(tmp_path)
    cfg["optim"] = {"amp": "fp16"}
    payload = _payload(model, cfg)

    with pytest.raises(HeatmapControlResumeError, match="scaler_state_dict"):
        validate_heatmap_control_resume_checkpoint(
            _write_resume(tmp_path, payload, "missing-scaler.pth"), model, cfg
        )

    payload["scaler_state_dict"] = {"scale": 65536.0}
    report = validate_heatmap_control_resume_checkpoint(
        _write_resume(tmp_path, payload, "with-scaler.pth"), model, cfg
    )
    assert report["validated_scaler_state"] is True


def test_strict_generic_loader_raises_instead_of_skipping_bad_optimizer(
    tmp_path: Path,
) -> None:
    model = nn.Linear(2, 2)
    optimizer = torch.optim.AdamW(model.parameters())
    payload = {
        "trainable_state_dict": {
            name: value.detach().clone()
            for name, value in model.state_dict().items()
        },
        "optimizer_state_dict": {"state": {}, "param_groups": []},
    }
    path = _write_resume(tmp_path, payload, "bad-optimizer.pth")

    with pytest.raises(RuntimeError, match="Strict resume failed to restore Optimizer"):
        load_checkpoint_for_resume(
            str(path),
            model,
            optimizer=optimizer,
            strict_state_restore=True,
        )


@pytest.mark.parametrize("failure", ["missing", "shape", "nonfinite", "nontensor"])
def test_trainable_state_tensor_contract(tmp_path: Path, failure: str) -> None:
    model = _ControlModel()
    cfg = _base_config(tmp_path)
    payload = _payload(model, cfg)
    state = payload["trainable_state_dict"]
    name = next(iter(state))
    if failure == "missing":
        state.pop(name)
        match = "missing="
    elif failure == "shape":
        state[name] = torch.ones(1)
        match = "shape mismatch"
    elif failure == "nonfinite":
        state[name].view(-1)[0] = float("nan")
        match = "non-finite"
    else:
        state[name] = "not-a-tensor"
        match = "is not a tensor"

    with pytest.raises(HeatmapControlResumeError, match=match):
        validate_heatmap_control_resume_checkpoint(
            _write_resume(tmp_path, payload), model, cfg
        )


def test_duplicate_after_module_normalization_is_rejected(tmp_path: Path) -> None:
    model = _ControlModel()
    cfg = _base_config(tmp_path)
    payload = _payload(model, cfg)
    state = payload["trainable_state_dict"]
    name = next(iter(state))
    state["module." + name] = state[name].clone()

    with pytest.raises(HeatmapControlResumeError, match="duplicate keys"):
        validate_heatmap_control_resume_checkpoint(
            _write_resume(tmp_path, payload), model, cfg
        )


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda cfg: cfg["model"]["llm"].__setitem__("use_lora", True), "LoRA"),
        (
            lambda cfg: cfg["model"]["action_head"]["nextdit"].__setitem__(
                "internnav_system1_path", "/models/override.pth"
            ),
            "System1 overrides",
        ),
        (
            lambda cfg: cfg["runtime"].__setitem__(
                "base_checkpoint", "/models/inferred-base.pth"
            ),
            "inferred/warm-start",
        ),
        (
            lambda cfg: cfg["model"]["llm"].__setitem__(
                "model_path", "/models/different-internnav"
            ),
            "unified original InternNav path",
        ),
    ],
)
def test_saved_config_rejects_lora_native_overrides_and_inferred_base(
    tmp_path: Path, mutation, message: str
) -> None:
    model = _ControlModel()
    cfg = _base_config(tmp_path)
    payload = _payload(model, cfg)
    mutation(payload["config"])

    with pytest.raises(HeatmapControlResumeError, match=message):
        validate_heatmap_control_resume_checkpoint(
            _write_resume(tmp_path, payload), model, cfg
        )


def test_saved_dependency_and_control_architecture_must_match_current(tmp_path: Path) -> None:
    model = _ControlModel()
    cfg = _base_config(tmp_path)
    bad_dependency = _payload(model, cfg)
    bad_dependency["config"]["runtime"]["frozen_heatmap_dependency"][
        "tensor_count"
    ] = 9
    with pytest.raises(HeatmapControlResumeError, match="frozen_heatmap_dependency"):
        validate_heatmap_control_resume_checkpoint(
            _write_resume(tmp_path, bad_dependency, "bad-dependency.pth"), model, cfg
        )

    bad_architecture = _payload(model, cfg)
    bad_architecture["config"]["model"]["action_head"]["nextdit"][
        "heatmap_control"
    ]["age_normalizer_steps"] = 99.0
    with pytest.raises(HeatmapControlResumeError, match="architecture differs"):
        validate_heatmap_control_resume_checkpoint(
            _write_resume(tmp_path, bad_architecture, "bad-architecture.pth"), model, cfg
        )


@pytest.mark.parametrize(
    ("path", "value"),
    [
        ("data.mixture.epoch_size", 72160),
        ("data.mixture.seed", 99),
        (
            "data.trajectory_dagger.collection_roots",
            [f"/datasets/other/shard_{index:02d}" for index in range(4)],
        ),
        (
            "data.trajectory_dagger.expected_policy_fingerprint",
            "internnav-native-v1:" + "c" * 64,
        ),
        ("optim.batch_size", 2),
        ("optim.grad_accum_steps", 8),
        ("optim.heatmap_tokenizer_lr", 2.0e-4),
        ("optim.heatmap_control_lr", 2.0e-5),
        ("optim.heatmap_gate_lr", 2.0e-4),
        ("optim.scheduler", "linear"),
        ("optim.warmup_ratio", 0.1),
        ("optim.ema_decay", 0.99),
        ("optim.ema_warmup_steps", 100),
        ("training.stages.0.epochs", 4),
    ],
)
def test_exact_mixture_resume_rejects_training_contract_drift(
    tmp_path: Path,
    path: str,
    value: object,
) -> None:
    model = _ControlModel()
    cfg = _base_config(tmp_path)
    _enable_exact_mixture_resume(cfg)
    payload = _payload(model, cfg)
    current = copy.deepcopy(cfg)
    _set_nested(current, path, value)

    with pytest.raises(HeatmapControlResumeError, match="exact-resume"):
        validate_heatmap_control_resume_checkpoint(
            _write_resume(tmp_path, payload), model, current
        )


def test_exact_mixture_resume_requires_matching_sampler_state(tmp_path: Path) -> None:
    model = _ControlModel()
    cfg = _base_config(tmp_path)
    _enable_exact_mixture_resume(cfg)
    payload = _payload(model, cfg)

    report = validate_heatmap_control_resume_checkpoint(
        _write_resume(tmp_path, payload, "valid-mixture.pth"), model, cfg
    )
    assert report["validated_exact_training_contract"] is True
    assert report["validated_mixture_sampler_state"] is True

    missing = _payload(model, cfg)
    del missing["mixture_sampler_state"]
    with pytest.raises(HeatmapControlResumeError, match="mixture_sampler_state"):
        validate_heatmap_control_resume_checkpoint(
            _write_resume(tmp_path, missing, "missing-sampler.pth"), model, cfg
        )

    mismatched = _payload(model, cfg)
    mismatched["mixture_sampler_state"]["seed"] = 17
    with pytest.raises(HeatmapControlResumeError, match="sampler state"):
        validate_heatmap_control_resume_checkpoint(
            _write_resume(tmp_path, mismatched, "bad-sampler.pth"), model, cfg
        )


def test_exact_mixture_resume_requires_locked_native_dependency(
    tmp_path: Path,
) -> None:
    model = _ControlModel()
    cfg = _base_config(tmp_path)
    _enable_exact_mixture_resume(cfg)
    payload = _payload(model, cfg)
    cfg["runtime"]["native_internnav_dependency"]["manifest_sha256"] = "c" * 64

    with pytest.raises(HeatmapControlResumeError, match="released InternNav closure"):
        validate_heatmap_control_resume_checkpoint(
            _write_resume(tmp_path, payload, "bad-native-closure.pth"), model, cfg
        )


def test_mid_epoch_resume_requires_absolute_accumulation_aligned_batch(
    tmp_path: Path,
) -> None:
    model = _ControlModel()
    cfg = _base_config(tmp_path)
    _enable_exact_mixture_resume(cfg)
    payload = _payload(model, cfg)
    payload["batch"] = 1002
    with pytest.raises(HeatmapControlResumeError, match="accumulation_boundary"):
        validate_heatmap_control_resume_checkpoint(
            _write_resume(tmp_path, payload), model, cfg
        )

    epoch_end = _payload(model, cfg)
    epoch_end["batch"] = 9000
    with pytest.raises(HeatmapControlResumeError, match="strictly before"):
        validate_heatmap_control_resume_checkpoint(
            _write_resume(tmp_path, epoch_end, "epoch-end-mid.pth"), model, cfg
        )


def test_load_weights_is_forbidden_only_for_control(tmp_path: Path) -> None:
    cfg = _base_config(tmp_path)
    with pytest.raises(HeatmapControlResumeError, match="forbids --load-weights"):
        reject_heatmap_control_load_weights(cfg, "/models/arbitrary.pth")

    cfg["model"]["action_head"]["nextdit"]["heatmap_control"]["enabled"] = False
    reject_heatmap_control_load_weights(cfg, "/models/legacy-stage.pth")


def test_train_calls_guard_before_every_generic_resume_load() -> None:
    source_path = Path(__file__).parents[1] / "scripts" / "train.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    calls: dict[str, list[int]] = {
        "validate_heatmap_control_resume_checkpoint": [],
        "load_checkpoint_for_resume": [],
    }
    loader_calls: list[ast.Call] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
            if node.func.id in calls:
                calls[node.func.id].append(node.lineno)
            if node.func.id == "load_checkpoint_for_resume":
                loader_calls.append(node)

    validators = sorted(calls["validate_heatmap_control_resume_checkpoint"])
    loaders = sorted(calls["load_checkpoint_for_resume"])
    assert len(validators) == len(loaders) == 2
    assert all(guard < loader for guard, loader in zip(validators, loaders, strict=True))
    assert len(loader_calls) == 2
    assert all(
        any(
            keyword.arg == "strict_state_restore"
            and isinstance(keyword.value, ast.Name)
            and keyword.value.id == "heatmap_control_resume_guard"
            for keyword in call.keywords
        )
        for call in loader_calls
    )

    main_source = ast.get_source_segment(
        source_path.read_text(encoding="utf-8"),
        next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == "main"),
    )
    assert main_source is not None
    assert main_source.index("reject_heatmap_control_load_weights") < main_source.index(
        "_infer_base_checkpoint_from_resume"
    )
    assert "and not heatmap_control_resume_guard" in main_source

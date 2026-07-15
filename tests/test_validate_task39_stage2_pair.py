from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import torch
from scripts.tools import validate_task39_stage2_pair as validator


@pytest.fixture(autouse=True)
def _small_adapter_contract(monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(validator, "EXPECTED_ADAPTER_NUMEL", 10)
    monkeypatch.setattr(
        validator,
        "EXPECTED_ADAPTER_LAYOUT",
        {
            "mlp.0.weight": ((4,), torch.float32),
            "mlp.0.bias": ((2,), torch.float32),
            "mlp.3.weight": ((3,), torch.float32),
            "mlp.3.bias": ((1,), torch.float32),
        },
    )


def _adapter_state(offset: float = 0.0) -> dict[str, torch.Tensor]:
    return {
        "mlp.0.weight": torch.arange(4, dtype=torch.float32) + offset,
        "mlp.0.bias": torch.arange(2, dtype=torch.float32) + offset,
        "mlp.3.weight": torch.arange(3, dtype=torch.float32) + offset,
        "mlp.3.bias": torch.arange(1, dtype=torch.float32) + offset,
    }


def _write_file(path: Path, payload: bytes = b"fixture\n") -> Path:
    path.write_bytes(payload)
    return path


def _make_pair(
    tmp_path: Path,
    *,
    warm_args_update: dict[str, Any] | None = None,
    control_args_update: dict[str, Any] | None = None,
    warm_checkpoint_update: dict[str, Any] | None = None,
    control_checkpoint_update: dict[str, Any] | None = None,
    warm_state: dict[str, torch.Tensor] | None = None,
    control_state: dict[str, torch.Tensor] | None = None,
    warm_split_bytes: bytes = b'{"train_indices":[1,2],"val_indices":[3]}\n',
    control_split_bytes: bytes | None = None,
) -> dict[str, Path]:
    warm_base = _write_file(tmp_path / "warm_base.pth", b"warm base\n")
    control_base = _write_file(tmp_path / "control_base.pth", b"control base\n")
    shared_init = _write_file(tmp_path / "shared_init.pth", b"shared init\n")
    teacher = _write_file(tmp_path / "teacher.jsonl", b"teacher\n")

    common_args: dict[str, Any] = {
        **validator.EXPECTED_FIXED_ARGS,
        "resume_adapter": str(shared_init),
        "teacher_jsonl": str(teacher),
        "student_config": "configs/train_pano_adapter_stage2_8gpu.yaml",
        "adapter_config": "configs/adapter_pano_stage2_h1024.yaml",
    }
    warm_args = {
        **common_args,
        "base_checkpoint": str(warm_base),
        "output_dir": str(tmp_path / "warm_output"),
    }
    control_args = {
        **common_args,
        "base_checkpoint": str(control_base),
        "output_dir": str(tmp_path / "control_output"),
    }
    if warm_args_update:
        warm_args.update(warm_args_update)
    if control_args_update:
        control_args.update(control_args_update)

    warm_payload: dict[str, Any] = {
        "adapter_type": "pano_latent_space",
        "adapter_state_dict": _adapter_state() if warm_state is None else warm_state,
        "optimizer_state_dict": {},
        "epoch": 3,
        "step": 100,
        "args": warm_args,
    }
    control_payload: dict[str, Any] = {
        "adapter_type": "pano_latent_space",
        "adapter_state_dict": _adapter_state(1.0) if control_state is None else control_state,
        "optimizer_state_dict": {},
        "epoch": 3,
        "step": 100,
        "args": control_args,
    }
    if warm_checkpoint_update:
        warm_payload.update(warm_checkpoint_update)
    if control_checkpoint_update:
        control_payload.update(control_checkpoint_update)

    warm_checkpoint = tmp_path / "warm_epoch_003.pth"
    control_checkpoint = tmp_path / "control_epoch_003.pth"
    torch.save(warm_payload, warm_checkpoint)
    torch.save(control_payload, control_checkpoint)
    warm_split = _write_file(tmp_path / "warm_split.json", warm_split_bytes)
    control_split = _write_file(
        tmp_path / "control_split.json",
        warm_split_bytes if control_split_bytes is None else control_split_bytes,
    )
    return {
        "warm_checkpoint": warm_checkpoint,
        "control_checkpoint": control_checkpoint,
        "warm_split": warm_split,
        "control_split": control_split,
        "warm_base": warm_base,
        "control_base": control_base,
        "shared_init": shared_init,
        "teacher": teacher,
    }


def _validate(pair: dict[str, Path], *, output_json: Path | None = None) -> dict[str, Any]:
    return validator.validate_task39_stage2_pair(
        pair["warm_checkpoint"],
        pair["control_checkpoint"],
        pair["warm_split"],
        pair["control_split"],
        pair["warm_base"],
        pair["control_base"],
        pair["shared_init"],
        output_json=output_json,
    )


def test_valid_pair_reports_strong_hashes_and_writes_json(tmp_path: Path):
    pair = _make_pair(tmp_path)
    output = tmp_path / "reports" / "stage2_pair.json"

    summary = _validate(pair, output_json=output)
    written = json.loads(output.read_text(encoding="utf-8"))
    warm_payload = torch.load(pair["warm_checkpoint"], map_location="cpu", weights_only=False)
    control_payload = torch.load(pair["control_checkpoint"], map_location="cpu", weights_only=False)

    assert summary == written
    assert summary["schema"] == validator.SUMMARY_SCHEMA
    assert summary["passed"] is True
    assert all(summary["checks"].values())
    assert summary["contract"]["adapter_numel"] == 10
    assert summary["contract"]["adapter_dtype"] == "torch.float32"
    assert summary["contract"]["fixed_training_args"] == validator.EXPECTED_FIXED_ARGS
    assert summary["args_comparison_excluded_keys"] == ["base_checkpoint", "output_dir"]
    assert summary["arms"]["warm"]["checkpoint_file_sha256"] == validator.file_sha256(pair["warm_checkpoint"])
    assert summary["arms"]["control"]["checkpoint_file_sha256"] == validator.file_sha256(pair["control_checkpoint"])
    assert summary["arms"]["warm"]["adapter_state_sha256"] == validator.tensor_state_sha256(
        warm_payload["adapter_state_dict"]
    )
    assert summary["arms"]["control"]["adapter_state_sha256"] == validator.tensor_state_sha256(
        control_payload["adapter_state_dict"]
    )
    assert summary["arms"]["warm"]["split_sha256"] == summary["split_sha256"]
    assert summary["arms"]["control"]["split_sha256"] == summary["split_sha256"]
    assert summary["teacher_sidecar_path"] == str(pair["teacher"].resolve())


@pytest.mark.parametrize(
    ("checkpoint_update", "args_update", "match"),
    [
        ({"epoch": 2}, None, "epoch=2"),
        ({"step": 0}, None, "positive integer"),
        ({"adapter_type": "wrong"}, None, "adapter_type"),
        (None, {"adapter_hidden_dim": 256}, "adapter_hidden_dim"),
    ],
)
def test_rejects_invalid_checkpoint_metadata(
    tmp_path: Path,
    checkpoint_update: dict[str, Any] | None,
    args_update: dict[str, Any] | None,
    match: str,
):
    pair = _make_pair(
        tmp_path,
        warm_checkpoint_update=checkpoint_update,
        warm_args_update=args_update,
    )

    with pytest.raises(validator.Stage2PairValidationError, match=match):
        _validate(pair)


def test_rejects_wrong_adapter_tensor_count(tmp_path: Path):
    state = _adapter_state()
    state.pop("mlp.3.bias")
    pair = _make_pair(tmp_path, warm_state=state)

    with pytest.raises(validator.Stage2PairValidationError, match="tensor count=3"):
        _validate(pair)


def test_rejects_different_optimizer_step_counts(tmp_path: Path):
    pair = _make_pair(
        tmp_path,
        control_checkpoint_update={"step": 101},
    )

    with pytest.raises(validator.Stage2PairValidationError, match="optimizer-step counts differ"):
        _validate(pair)


def test_rejects_wrong_adapter_numel(tmp_path: Path):
    state = _adapter_state()
    state["mlp.3.bias"] = torch.zeros(2)
    pair = _make_pair(tmp_path, warm_state=state)

    with pytest.raises(validator.Stage2PairValidationError, match="adapter numel=11"):
        _validate(pair)


def test_rejects_wrong_adapter_tensor_names(tmp_path: Path):
    state = _adapter_state()
    state["wrong.weight"] = state.pop("mlp.0.weight")
    pair = _make_pair(tmp_path, warm_state=state)

    with pytest.raises(validator.Stage2PairValidationError, match="tensor names mismatch"):
        _validate(pair)


def test_rejects_wrong_adapter_tensor_shape_even_when_numel_matches(tmp_path: Path):
    state = _adapter_state()
    state["mlp.0.weight"] = state["mlp.0.weight"].reshape(2, 2)
    pair = _make_pair(tmp_path, warm_state=state)

    with pytest.raises(validator.Stage2PairValidationError, match=r"shape=.*expected"):
        _validate(pair)


def test_rejects_non_float32_adapter_tensor(tmp_path: Path):
    state = _adapter_state()
    state["mlp.3.weight"] = state["mlp.3.weight"].to(torch.bfloat16)
    pair = _make_pair(tmp_path, warm_state=state)

    with pytest.raises(validator.Stage2PairValidationError, match=r"dtype=torch\.bfloat16"):
        _validate(pair)


@pytest.mark.parametrize("bad_value", [float("nan"), float("inf"), -float("inf")])
def test_rejects_nonfinite_adapter_state(tmp_path: Path, bad_value: float):
    state = _adapter_state()
    state["mlp.0.weight"][0] = bad_value
    pair = _make_pair(tmp_path, warm_state=state)

    with pytest.raises(validator.Stage2PairValidationError, match="non-finite"):
        _validate(pair)


@pytest.mark.parametrize(
    ("key", "value", "match"),
    [
        ("seed", 43, "args.seed"),
        ("seed", 42.0, "args.seed"),
        ("epochs", 2, "args.epochs"),
        ("batch_size", 8, "args.batch_size"),
        ("lr", 1.0e-4, "args.lr"),
        ("weight_decay", 0.02, "args.weight_decay"),
        ("grad_clip", 0.5, "args.grad_clip"),
        ("adapter_dropout", 0.1, "args.adapter_dropout"),
        ("max_samples", 16, "args.max_samples"),
        ("dataset_max_clips", 2, "args.dataset_max_clips"),
        ("val_ratio", 0.2, "args.val_ratio"),
        ("prefetch_batches", 4, "args.prefetch_batches"),
        ("prefetch_workers", 1, "args.prefetch_workers"),
        ("startup_preflight_batches", 0, "args.startup_preflight_batches"),
        ("teacher_target_mode", "aligned", "args.teacher_target_mode"),
        ("compute_teacher_mse", True, "args.compute_teacher_mse"),
        ("teacher_cache_mode", "none", "args.teacher_cache_mode"),
        ("teacher_preload_cache", False, "args.teacher_preload_cache"),
        ("teacher_preload_workers", 1, "args.teacher_preload_workers"),
        ("check_teacher_tensor_files", True, "args.check_teacher_tensor_files"),
        ("raw_distill_weight", 0.0, "args.raw_distill_weight"),
        ("cond_distill_weight", 0.0, "args.cond_distill_weight"),
        ("gt_weight", 0.0, "args.gt_weight"),
        ("save_every_epochs", 0, "args.save_every_epochs"),
    ],
)
def test_rejects_shared_wrong_fixed_training_contract(
    tmp_path: Path,
    key: str,
    value: Any,
    match: str,
):
    pair = _make_pair(
        tmp_path,
        warm_args_update={key: value},
        control_args_update={key: value},
    )

    with pytest.raises(validator.Stage2PairValidationError, match=match):
        _validate(pair)


def test_rejects_missing_fixed_training_arg(tmp_path: Path):
    pair = _make_pair(tmp_path)
    for checkpoint_name in ("warm_checkpoint", "control_checkpoint"):
        payload = torch.load(pair[checkpoint_name], map_location="cpu", weights_only=False)
        del payload["args"]["teacher_cache_mode"]
        torch.save(payload, pair[checkpoint_name])

    with pytest.raises(validator.Stage2PairValidationError, match=r"args\.teacher_cache_mode is missing"):
        _validate(pair)


def test_rejects_any_unregistered_arg_difference(tmp_path: Path):
    pair = _make_pair(tmp_path, control_args_update={"prefetch_batches": 4})

    with pytest.raises(validator.Stage2PairValidationError, match="prefetch_batches"):
        _validate(pair)


def test_rejects_expected_base_mismatch(tmp_path: Path):
    pair = _make_pair(tmp_path)
    wrong_base = _write_file(tmp_path / "wrong_base.pth")

    with pytest.raises(validator.Stage2PairValidationError, match="base_checkpoint mismatch"):
        validator.validate_task39_stage2_pair(
            pair["warm_checkpoint"],
            pair["control_checkpoint"],
            pair["warm_split"],
            pair["control_split"],
            wrong_base,
            pair["control_base"],
            pair["shared_init"],
        )


def test_rejects_non_shared_resume_path(tmp_path: Path):
    other_init = _write_file(tmp_path / "other_init.pth")
    pair = _make_pair(tmp_path, control_args_update={"resume_adapter": str(other_init)})

    with pytest.raises(validator.Stage2PairValidationError, match="expected shared init"):
        _validate(pair)


def test_rejects_different_teacher_sidecars(tmp_path: Path):
    other_teacher = _write_file(tmp_path / "other_teacher.jsonl")
    pair = _make_pair(tmp_path, control_args_update={"teacher_jsonl": str(other_teacher)})

    with pytest.raises(validator.Stage2PairValidationError, match="teacher sidecars differ"):
        _validate(pair)


def test_rejects_semantically_equal_but_not_byte_identical_splits(tmp_path: Path):
    pair = _make_pair(
        tmp_path,
        warm_split_bytes=b'{"train_indices":[1],"val_indices":[2]}\n',
        control_split_bytes=b'{ "train_indices": [1], "val_indices": [2] }\n',
    )

    with pytest.raises(validator.Stage2PairValidationError, match="not byte-identical"):
        _validate(pair)


def test_existing_output_is_never_overwritten(tmp_path: Path):
    pair = _make_pair(tmp_path)
    output = _write_file(tmp_path / "stage2_pair.json", b"sentinel\n")

    with pytest.raises(FileExistsError, match="Refusing to overwrite"):
        _validate(pair, output_json=output)

    assert output.read_bytes() == b"sentinel\n"


def test_cli_accepts_all_pair_inputs_and_prints_summary(tmp_path: Path, capsys: pytest.CaptureFixture[str]):
    pair = _make_pair(tmp_path)

    result = validator.main(
        [
            "--warm-checkpoint",
            str(pair["warm_checkpoint"]),
            "--control-checkpoint",
            str(pair["control_checkpoint"]),
            "--warm-split",
            str(pair["warm_split"]),
            "--control-split",
            str(pair["control_split"]),
            "--expected-warm-base",
            str(pair["warm_base"]),
            "--expected-control-base",
            str(pair["control_base"]),
            "--shared-init",
            str(pair["shared_init"]),
        ]
    )

    assert result == 0
    assert json.loads(capsys.readouterr().out)["passed"] is True

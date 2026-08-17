"""Static contracts for the formal eight-GPU/four-root control launcher."""

from __future__ import annotations

import copy
import re
import symtable
import subprocess
from pathlib import Path

import pytest
import yaml

from scripts.training.formal_heatmap_control_contract import (
    FormalHeatmapControlContractError,
    assert_formal_heatmap_control_no_training_eval,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
LAUNCHER = REPO_ROOT / "scripts/run_heatmap_system1_control_8gpu_mxc500.sh"
EVAL_ONLY_LAUNCHER = (
    REPO_ROOT
    / "scripts/run_heatmap_system1_control_epoch3_eval_only_8gpu_mxc500.sh"
)
CONFIG = REPO_ROOT / "configs/heatmap_system1_control_8gpu.yaml"


def test_train_main_does_not_shadow_heatmap_control_enabled_helper() -> None:
    source_path = REPO_ROOT / "scripts/train.py"
    source = source_path.read_text(encoding="utf-8")
    module_table = symtable.symtable(source, str(source_path), "exec")
    main_table = next(
        child
        for child in module_table.get_children()
        if child.get_name() == "main" and child.get_type() == "function"
    )
    symbol = main_table.lookup("heatmap_control_enabled")
    assert symbol.is_global()
    assert not symbol.is_local()


def test_launcher_is_eight_gpu_but_consumes_exactly_four_data_roots() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "readonly NUM_GPUS=8" in source
    assert "readonly NUM_DAGGER_ROOTS=4" in source
    assert "full_train_4way_seed17/training_roots.json" in source
    assert "--nproc_per_node=8" in source
    assert 'partition.get("num_shards") == expected_root_count' in source
    assert "for index in $(seq 0 $((NUM_DAGGER_ROOTS - 1)))" in source

    config = CONFIG.read_text(encoding="utf-8")
    roots = re.findall(r"\$\{DAGGER_ROOT_(\d\d)\}", config)
    assert roots == ["00", "01", "02", "03"]


def test_launcher_and_config_lock_exact_epoch_and_order_contract() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    config = CONFIG.read_text(encoding="utf-8")
    assert 'HEATMAP_CONTROL_EPOCH_SIZE="${HEATMAP_CONTROL_EPOCH_SIZE:-72000}"' in source
    assert "HEATMAP_CONTROL_EPOCH_SIZE == 72000" in source
    assert "HEATMAP_CONTROL_EPOCH_SIZE % 160 == 0" in source
    assert 'assert data["mixture"]["epoch_size"] == expected_epoch_size' in source
    assert 'assert data["in_order"] is True' in source
    assert 'assert gpu["devices"] == list(range(8))' in source
    assert "in_order: true" in config
    assert "epoch_size: 72000" in config


def test_formal_config_disables_all_train_side_evaluation_and_best_selection() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    contract = assert_formal_heatmap_control_no_training_eval(
        config,
        require_formal_recipe=True,
    )
    assert contract["per_epoch_validation"] is False
    assert contract["pre_training_validation"] is False
    assert contract["best_checkpoint_selection"] is False
    assert contract["external_eval_checkpoint"] == "epoch_003.pth"
    assert "assert_formal_heatmap_control_no_training_eval" in source
    assert 'require_formal_recipe=True' in source
    assert 'stage["epochs"] == 3' in source


@pytest.mark.parametrize(
    ("section", "key", "value"),
    [
        ("validation", "enabled", True),
        ("validation", "eval_every_epochs", 1),
        ("validation", "best_selection_enabled", True),
        ("validation", "evaluate_before_training", True),
        ("validation", "baseline_as_best_threshold", True),
        ("validation", "val_inference_batches", 1),
        ("log", "save_every_epochs", 2),
        ("log", "val_vis_batches", 1),
        ("data", "val_root", "/alternate/validation-data"),
    ],
)
def test_alternate_formal_config_cannot_enable_train_side_evaluation(
    section: str,
    key: str,
    value: object,
) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    alternate = copy.deepcopy(config)
    alternate[section][key] = value
    with pytest.raises(
        FormalHeatmapControlContractError,
        match="train-side evaluation contract violated",
    ):
        assert_formal_heatmap_control_no_training_eval(
            alternate,
            require_formal_recipe=True,
        )


@pytest.mark.parametrize(("key", "value"), [("epochs", 4), ("name", "alternate")])
def test_alternate_formal_stage_cannot_bypass_three_epoch_contract(
    key: str,
    value: object,
) -> None:
    config = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    config["training"]["stages"][0][key] = value
    with pytest.raises(FormalHeatmapControlContractError):
        assert_formal_heatmap_control_no_training_eval(
            config,
            require_formal_recipe=True,
        )


def test_launcher_hashes_native_model_once_and_train_persists_closure() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    train_source = (REPO_ROOT / "scripts/train.py").read_text(encoding="utf-8")
    assert (
        "internnav_native_r2r_val_unseen_8gpu_20260802/manifests/"
        "internnav_model.sha256"
    ) in source
    assert (
        'EXPECTED_NATIVE_MODEL_MANIFEST_SHA256="'
        "f37a6df2e0703e38c34ccdba89c861bb8490ad3a36201bc1ec24a7509bf56581"
        '"'
    ) in source
    assert "readonly EXPECTED_NATIVE_MODEL_FILE_COUNT=14" in source
    assert source.count('sha256sum -c "$NATIVE_MODEL_MANIFEST"') == 1
    assert 'export HEATMAPVLN_NATIVE_MODEL_VERIFIED=1' in source
    assert 'export HEATMAPVLN_NATIVE_MODEL_FILE_COUNT="$EXPECTED_NATIVE_MODEL_FILE_COUNT"' in source
    injection = train_source.index(
        "inject_native_internnav_dependency_from_env(cfg)"
    )
    ddp_init = train_source.index("dist_context = init_distributed_context(cfg)")
    assert injection < ddp_init


def test_launcher_rejects_resume_during_dry_run_and_is_valid_bash() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "dry-run cannot be combined with HEATMAP_CONTROL_AUTO_RESUME=1" in source
    assert "dry-run cannot be combined with HEATMAP_CONTROL_RESUME" in source
    result = subprocess.run(
        ["bash", "-n", str(LAUNCHER)],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_launcher_hands_complete_epoch_three_ema_to_full_eight_gpu_eval() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    assert "HEATMAP_CONTROL_AUTO_EVAL=\"${HEATMAP_CONTROL_AUTO_EVAL:-1}\"" in source
    assert "latest/checkpoints/epoch_003.pth" in source
    assert "validate_heatmap_control_deployment_checkpoint" in source
    assert "checkpoints/best.pth" not in source
    assert "epoch_003_batch" not in source
    assert 'EVAL_GPU_DEVICES="${EVAL_GPU_DEVICES:-$GPU_DEVICES}"' in source
    assert '[[ "$EVAL_GPU_DEVICES" == "$GPU_DEVICES" ]]' in source
    assert "EVAL_X11_MODE=bundle" in source
    assert (
        "unset EVAL_PREFLIGHT_ONLY EVAL_SMOKE_ONLY EVAL_SKIP_SMOKE "
        "EVAL_REUSE_XVFB"
    ) in source
    assert "${FINAL_CHECKPOINT_SHA256:0:12}" in source
    assert 'CONTROL_EVAL_SERVER_SHA256="$(sha256sum -- "$CONTROL_EVAL_SERVER"' in source
    assert "_plan${CONTROL_EVAL_SERVER_SHA256:0:12}" in source
    assert "run_8gpu_heatmap_control_rpc_eval.sh" in source
    assert 'bash "$CONTROL_EVAL_LAUNCHER"' in source


def test_launcher_holds_an_exclusive_output_lock_through_eval() -> None:
    source = LAUNCHER.read_text(encoding="utf-8")
    lock = source.index("TRAIN_EVAL_LOCK=")
    acquire = source.index("flock -n 9")
    train = source.index('CUDA_VISIBLE_DEVICES="$GPU_DEVICES" "$TORCHRUN"')
    evaluate = source.index('bash "$CONTROL_EVAL_LAUNCHER"')
    assert lock < acquire < train < evaluate
    assert ".heatmap_system1_control_train_eval.lock" in source


def test_epoch_three_eval_only_launcher_never_reenters_training() -> None:
    source = EVAL_ONLY_LAUNCHER.read_text(encoding="utf-8")
    assert "run_20260807_112540/checkpoints/epoch_003.pth" in source
    assert "a556329887be4e6d33f129e1bc670c6515d6a3634b2f3a210ff40b8d21dc9635" in source
    assert "run_8gpu_heatmap_control_rpc_eval.sh" in source
    assert "EVAL_CONTROL_MODE=on" in source
    assert "EVAL_SMOKE_ONLY EVAL_SKIP_SMOKE" in source
    assert "torchrun" not in source
    assert "scripts/train.py" not in source
    result = subprocess.run(
        ["bash", "-n", str(EVAL_ONLY_LAUNCHER)],
        check=False,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr


def test_train_uses_in_order_loader_and_absolute_mid_epoch_batches() -> None:
    train_source = (REPO_ROOT / "scripts/train.py").read_text(encoding="utf-8")
    loop_source = (
        REPO_ROOT / "scripts/training/train_loop.py"
    ).read_text(encoding="utf-8")
    assert "in_order=train_loader_in_order" in train_source
    assert "not num_workers > 0" in train_source
    assert "train_sampler.load_state_dict(resume_mixture_sampler_state)" in train_source
    assert "completed_epoch_batches = i + 1" in loop_source
    assert "completed_epoch_batches < len(train_loader)" in loop_source
    assert "batch=completed_epoch_batches" in loop_source
    assert 'mid_extra_state["mixture_sampler_state"] = sampler_state' in loop_source


def test_train_loop_never_evaluates_or_selects_best_when_formal_flags_are_off() -> None:
    source = (REPO_ROOT / "scripts/train.py").read_text(encoding="utf-8")
    epoch_override_guard = source.index("if args.epochs not in (None, 3):")
    epoch_override_application = source.index("if args.epochs is not None:")
    assert epoch_override_guard < epoch_override_application
    assert "formal heatmap-control training forbids --epochs overrides" in source
    assert "validation_enabled\n            and val_loader is not None" in source
    assert "if best_selection_enabled:" in source
    assert "if best_selection_enabled and no_improve_count >= patience:" in source
    assert "checkpoint_selection_state=(" in source
    assert "if best_selection_enabled\n                else None" in source
    assert "epoch % cfg['log']['save_every_epochs'] == 0 or is_best" in source

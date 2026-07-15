from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest
import yaml
from scripts.train import _dataloader_in_order_kwargs

_REPO_ROOT = Path(__file__).resolve().parents[1]
_STAGE3_CONFIG = _REPO_ROOT / "configs/train_stage3_pano_system1_h1024_8gpu.yaml"
_STAGE3_LAUNCHER = (
    _REPO_ROOT / "scripts/run_stage3_pano_system1_h1024_8gpu_mxc500_launcher.sh"
)


def _render_launcher_config(tmp_path, *, configured: bool, env_override: str | None) -> dict:
    config = yaml.safe_load(_STAGE3_CONFIG.read_text())
    config["data"]["in_order"] = configured
    base_config = tmp_path / "base.yaml"
    output_config = tmp_path / "output.yaml"
    base_config.write_text(yaml.safe_dump(config, sort_keys=False))

    launcher = _STAGE3_LAUNCHER.read_text()
    marker = 'python - "$STAGE3_CONFIG" "$TMP_CONFIG" <<\'PY\'\n'
    overlay_script = launcher.split(marker, maxsplit=1)[1].split("\nPY\n", maxsplit=1)[0]
    env = os.environ.copy()
    env.update(
        {
            "PANORAMIC_DATA_ROOT": "/data",
            "STAGE3_OUT_DIR": "/output",
            "STAGE3_TB_DIR": "/tensorboard",
            "INTERNNAV_MODEL_PATH": "/model",
            "STAGE3_ADAPTER_CKPT": "/adapter.pth",
            "GPU_DEVICES": "0,1",
        }
    )
    if env_override is None:
        env.pop("STAGE3_IN_ORDER", None)
    else:
        env["STAGE3_IN_ORDER"] = env_override

    subprocess.run(
        [sys.executable, "-", str(base_config), str(output_config)],
        input=overlay_script,
        text=True,
        env=env,
        check=True,
    )
    return yaml.safe_load(output_config.read_text())


@pytest.mark.parametrize(
    ("configured", "expected"),
    [
        (True, True),
        (False, False),
        ("true", True),
        ("false", False),
    ],
)
def test_train_dataloader_honors_explicit_in_order(configured, expected) -> None:
    cfg = {"data": {"in_order": configured}}
    assert _dataloader_in_order_kwargs(cfg, 4) == {"in_order": expected}


def test_train_dataloader_default_preserves_unordered_worker_behavior() -> None:
    assert _dataloader_in_order_kwargs({"data": {}}, 4) == {"in_order": False}


def test_zero_worker_loader_omits_in_order_keyword() -> None:
    assert _dataloader_in_order_kwargs({"data": {"in_order": True}}, 0) == {}


def test_validation_default_remains_ordered() -> None:
    assert _dataloader_in_order_kwargs(
        {"data": {"in_order": False}},
        4,
        validation=True,
    ) == {"in_order": True}


@pytest.mark.parametrize(
    ("configured", "env_override", "expected"),
    [
        (False, None, False),
        (False, "true", True),
        (True, "false", False),
    ],
)
def test_stage3_launcher_layers_optional_in_order_override(
    tmp_path,
    configured,
    env_override,
    expected,
) -> None:
    rendered = _render_launcher_config(
        tmp_path,
        configured=configured,
        env_override=env_override,
    )
    assert rendered["data"]["in_order"] is expected


def test_stage3_config_default_and_wrapper_preserve_caller_override() -> None:
    config = yaml.safe_load(_STAGE3_CONFIG.read_text())
    launcher = _STAGE3_LAUNCHER.read_text()
    wrapper = (
        _REPO_ROOT / "scripts/run_stage3_after_stage2_8gpu_mxc500.sh"
    ).read_text()

    assert config["data"]["in_order"] is False
    assert 'export STAGE3_IN_ORDER="${STAGE3_IN_ORDER:-}"' in launcher
    assert 'set_bool(data, "in_order", "STAGE3_IN_ORDER")' in launcher
    assert "STAGE3_IN_ORDER=" not in wrapper

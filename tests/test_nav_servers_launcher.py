"""scripts/deploy/start_nav_servers_cuda.sh: the attention switch, run against stand-in servers.

The launcher is driven end to end with NAV_PYTHON pointing at a shell script that plays both
servers and the launcher's own probes, so the test needs neither a GPU nor the model stack.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
NAV_LAUNCHER = REPO / "scripts" / "deploy" / "start_nav_servers_cuda.sh"
CUDA_LAUNCHER = REPO / "scripts" / "run_ppa_r2r_val_unseen_cuda.sh"

# Stands in for $NAV_PYTHON.  "-" runs the launcher's heredoc probes: the busy-port probe must
# find nobody listening (exit 1), the readiness probe succeeds once the servers have logged.
# "-u <server> ..." is a server: the model server records its argv and logs the lines the
# launcher greps for, confirming --attn_implementation unless FAKE_NO_CONFIRM is set.
FAKE_PYTHON = r"""#!/usr/bin/env bash
if [[ "$1" == "-" ]]; then
  src="$(cat)"
  case "$src" in
    *create_connection*) exit 1 ;;
    *VLAClient*) sleep 0.5; exit 0 ;;
  esac
  exit 0
fi
shift
server="$1"; shift
case "$server" in
  *rpc_model_server.py)
    printf '%s\n' "$@" > "$FAKE_OUT/model_argv"
    echo "Formal PPA online AMB3R runtime enabled"
    attn=""
    while [[ $# -gt 0 ]]; do
      [[ "$1" == "--attn_implementation" ]] && attn="$2"
      shift
    done
    [[ -n "$attn" && -z "${FAKE_NO_CONFIRM:-}" ]] && echo "Attention implementation confirmed: $attn"
    ;;
esac
sleep 2
"""


def _run(tmp_path: Path, **env_overrides: str) -> tuple[subprocess.CompletedProcess, list[str]]:
    root = tmp_path / "root"
    amb3r = root / "amb3r"
    for directory in (root / "rpc" / "src" / "vla_rpc", root / "internnav", amb3r / "slam", root / "da3"):
        directory.mkdir(parents=True)
    for file in (amb3r / "slam" / "slam_config.yaml", root / "da3" / "model.safetensors", root / "ckpt.pth"):
        file.write_text("x")
    python = tmp_path / "python"
    python.write_text(FAKE_PYTHON)
    python.chmod(0o755)
    out = tmp_path / "out"
    out.mkdir()
    env = {
        "PATH": os.environ.get("PATH", "/usr/bin:/bin"),
        "FAKE_OUT": str(out),
        "NAV_ROOT": str(root),
        "NAV_REPO": str(REPO),
        "NAV_RPC_ROOT": str(root / "rpc"),
        "INTERNNAV_MODEL_PATH": str(root / "internnav"),
        "NAV_AMB3R_ROOT": str(amb3r),
        "NAV_DA3_CHECKPOINT": str(root / "da3"),
        "NAV_CHECKPOINT": str(root / "ckpt.pth"),
        "NAV_PYTHON": str(python),
        "NAV_RUNTIME_DIR": str(tmp_path / "run"),
        "NAV_SERVER_START_TIMEOUT_S": "60",
        **env_overrides,
    }
    result = subprocess.run(
        ["bash", str(NAV_LAUNCHER)], env=env, capture_output=True, text=True, timeout=120
    )
    argv_file = out / "model_argv"
    argv = argv_file.read_text().split("\n") if argv_file.exists() else []
    return result, argv


def _attn_args(argv: list[str]) -> list[str]:
    return [argv[i + 1] for i, arg in enumerate(argv[:-1]) if arg == "--attn_implementation"]


pytestmark = pytest.mark.skipif(shutil.which("bash") is None, reason="needs bash")


def test_default_asks_the_model_server_for_flash_attention_and_checks_it_got_it(tmp_path):
    result, argv = _run(tmp_path)
    assert _attn_args(argv) == ["flash_attention_2"]
    assert "READY" in result.stdout and "attn=flash_attention_2" in result.stdout
    # The stand-in servers exit after 2 s; the launcher reports that and stops.
    assert result.returncode == 2 and "a server exited" in result.stderr


def test_sdpa_is_passed_through(tmp_path):
    result, argv = _run(tmp_path, NAV_ATTN_IMPL="sdpa")
    assert _attn_args(argv) == ["sdpa"]
    assert "attn=sdpa" in result.stdout


def test_a_server_that_does_not_confirm_the_attention_is_not_reported_ready(tmp_path):
    result, argv = _run(tmp_path, FAKE_NO_CONFIRM="1")
    assert _attn_args(argv) == ["flash_attention_2"]
    assert "READY" not in result.stdout
    assert result.returncode == 2 and "did not confirm attention flash_attention_2" in result.stderr


def test_same_default_and_mechanism_as_the_evaluation_launcher():
    """The robot path and the evaluation runs must not drift into different attentions."""
    nav = NAV_LAUNCHER.read_text(encoding="utf-8")
    cuda = CUDA_LAUNCHER.read_text(encoding="utf-8")
    assert 'ATTN_IMPL="${NAV_ATTN_IMPL:-flash_attention_2}"' in nav
    assert 'ATTN_IMPL="${PPA_EVAL_ATTN_IMPL:-flash_attention_2}"' in cuda
    line = '[[ -n "$ATTN_IMPL" ]] && MODEL_EXTRA+=(--attn_implementation "$ATTN_IMPL")'
    assert line in nav and line in cuda
    assert '--require_ppa_online_amb3r "${MODEL_EXTRA[@]}"' in nav

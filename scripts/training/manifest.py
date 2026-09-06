"""
Run metadata capture, file I/O helpers, and git state utilities.
"""

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml


def _make_json_safe(value: Any) -> Any:
    """Convert common training objects into JSON-serializable values."""
    if isinstance(value, dict):
        return {str(k): _make_json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_make_json_safe(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (np.floating, np.integer)):
        return value.item()
    if torch.is_tensor(value):
        if value.numel() == 1:
            return value.item()
        return value.detach().cpu().tolist()
    return value


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_make_json_safe(payload), f, indent=2, ensure_ascii=False)


def _write_yaml(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(payload, f, allow_unicode=True, sort_keys=False)


def _append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    record = {"timestamp": datetime.now().isoformat(), **_make_json_safe(payload)}
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


def _safe_symlink(link_path: Path, target: Any) -> None:
    if link_path.is_symlink() or link_path.exists():
        if link_path.is_dir() and not link_path.is_symlink():
            shutil.rmtree(link_path)
        else:
            link_path.unlink()
    link_path.symlink_to(target)


def _clear_directory(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for child in path.iterdir():
        if child.is_symlink() or child.is_file():
            child.unlink()
        elif child.is_dir():
            shutil.rmtree(child)


def _run_git_command(project_dir: Path, args: list[str], timeout_s: float = 5.0) -> str:
    try:
        result = subprocess.run(
            # The shared checkout is owned by another uid, so without
            # ``safe.directory`` git refuses every command and the manifest
            # silently records "no git" instead of the commit (2026-09-06).
            ["git", "-c", f"safe.directory={project_dir}", *args],
            cwd=project_dir,
            check=True,
            capture_output=True,
            text=True,
            timeout=timeout_s,
        )
        return result.stdout.strip()
    except Exception:
        return ""


def _capture_git_state(project_dir: Path) -> dict[str, Any]:
    commit = _run_git_command(project_dir, ["rev-parse", "HEAD"])
    short_commit = _run_git_command(project_dir, ["rev-parse", "--short", "HEAD"])
    branch = _run_git_command(project_dir, ["rev-parse", "--abbrev-ref", "HEAD"])
    status_short = _run_git_command(project_dir, ["status", "--short", "--untracked-files=no"])
    return {
        "commit": commit or None,
        "short_commit": short_commit or None,
        "branch": branch or None,
        "is_dirty": bool(status_short),
        "status_short": status_short.splitlines() if status_short else [],
    }


def _capture_env_state(
    args: argparse.Namespace,
    run_dir: Path,
    cfg: dict[str, Any],
    is_resuming: bool,
) -> dict[str, Any]:
    return {
        "run_dir": str(run_dir),
        "is_resuming": is_resuming,
        "argv": sys.argv,
        "config_path": args.config,
        "python": sys.version,
        "platform": platform.platform(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "torch_version": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version(),
        "device_count": torch.cuda.device_count() if torch.cuda.is_available() else 0,
        "configured_device": cfg.get("model", {}).get("device", "cuda"),
        "timestamp": datetime.now().isoformat(),
    }


def _find_resume_checkpoint(run_dir: Path) -> Path | None:
    candidates = [
        run_dir / "checkpoints" / "latest.pth",
        run_dir / "ckpts" / "latest.pth",
    ]
    for path in candidates:
        if path.exists():
            return path
    return None

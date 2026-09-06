"""The tree a cluster job actually ran is recorded, and a moved tree is refused.

2026-09-06: the EXP-14 job read the shared checkout while it was being edited.
The memory arm died on a half-applied change and the control arm's code version
is unprovable, because the run manifest recorded no commit at all.
"""
import subprocess
import sys
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.training.source_fingerprint import (  # noqa: E402
    SOURCE_FINGERPRINT_ENV,
    check_pinned_source_fingerprint,
    compute_source_fingerprint,
)


def _tree(tmp_path: Path) -> Path:
    (tmp_path / "src").mkdir()
    (tmp_path / "scripts").mkdir()
    (tmp_path / "configs").mkdir()
    (tmp_path / "src" / "a.py").write_text("x = 1\n")
    (tmp_path / "scripts" / "b.py").write_text("y = 2\n")
    (tmp_path / "configs" / "c.yaml").write_text("k: v\n")
    return tmp_path


def test_fingerprint_is_stable_and_counts_every_source(tmp_path):
    tree = _tree(tmp_path)
    first = compute_source_fingerprint(tree)
    assert first["file_count"] == 3
    assert first == compute_source_fingerprint(tree)


def test_editing_any_source_moves_the_fingerprint(tmp_path):
    tree = _tree(tmp_path)
    before = compute_source_fingerprint(tree)["fingerprint"]
    (tree / "src" / "a.py").write_text("x = 2\n")
    assert compute_source_fingerprint(tree)["fingerprint"] != before


def test_a_config_edit_moves_the_fingerprint(tmp_path):
    """Arm configs decide the experiment, so they are part of the code."""
    tree = _tree(tmp_path)
    before = compute_source_fingerprint(tree)["fingerprint"]
    (tree / "configs" / "c.yaml").write_text("k: w\n")
    assert compute_source_fingerprint(tree)["fingerprint"] != before


def test_bytecode_caches_are_ignored(tmp_path):
    tree = _tree(tmp_path)
    before = compute_source_fingerprint(tree)["fingerprint"]
    cache = tree / "src" / "__pycache__"
    cache.mkdir()
    (cache / "a.cpython-312.py").write_text("stale\n")
    assert compute_source_fingerprint(tree)["fingerprint"] == before


def test_unpinned_runs_are_allowed(monkeypatch):
    monkeypatch.delenv(SOURCE_FINGERPRINT_ENV, raising=False)
    check_pinned_source_fingerprint("anything")


def test_a_moved_tree_is_refused(monkeypatch):
    monkeypatch.setenv(SOURCE_FINGERPRINT_ENV, "pinned-by-the-launcher")
    check_pinned_source_fingerprint("pinned-by-the-launcher")
    with pytest.raises(RuntimeError, match="source tree changed"):
        check_pinned_source_fingerprint("something-else")


def test_the_cli_prints_the_same_value(tmp_path):
    tree = _tree(tmp_path)
    out = subprocess.run(
        [sys.executable, str(REPO / "scripts" / "tools" / "source_fingerprint.py"), str(tree)],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    assert out == compute_source_fingerprint(tree)["fingerprint"]

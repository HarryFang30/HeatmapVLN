"""Content hash of the source tree, for run provenance.

Cluster jobs read the shared checkout live, one arm after another, so a tree
edited mid-job hands later arms different code than earlier ones with nothing in
the run directory recording it.  A launcher pins the tree by exporting
``HEATMAPVLN_SOURCE_FINGERPRINT``; every arm started afterwards refuses to run if
the tree has moved.

Stdlib only on purpose: the blank container's preflight runs this before torch or
numpy are importable.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Any

#: Source roots whose content defines "the code this run executed".
SOURCE_FINGERPRINT_ROOTS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("src", (".py",)),
    ("scripts", (".py",)),
    ("configs", (".yaml", ".yml")),
)

#: Environment variable a launcher sets to pin the tree for a whole job.
SOURCE_FINGERPRINT_ENV = "HEATMAPVLN_SOURCE_FINGERPRINT"


def compute_source_fingerprint(project_dir: Path) -> dict[str, Any]:
    """Hash every tracked-by-convention source file under the project."""
    project_dir = Path(project_dir)
    digest = hashlib.sha256()
    counted = 0
    for root_name, suffixes in SOURCE_FINGERPRINT_ROOTS:
        root = project_dir / root_name
        if not root.is_dir():
            continue
        paths = sorted(
            path
            for path in root.rglob("*")
            if path.is_file()
            and path.suffix in suffixes
            and "__pycache__" not in path.parts
        )
        for path in paths:
            digest.update(str(path.relative_to(project_dir)).encode())
            digest.update(b"\0")
            digest.update(hashlib.sha256(path.read_bytes()).digest())
            counted += 1
    return {
        "fingerprint": digest.hexdigest(),
        "file_count": counted,
        "roots": [name for name, _ in SOURCE_FINGERPRINT_ROOTS],
    }


def check_pinned_source_fingerprint(observed: str) -> None:
    """Refuse to run when the tree moved since the launcher pinned it."""
    expected = os.environ.get(SOURCE_FINGERPRINT_ENV, "").strip()
    if not expected or expected == observed:
        return
    raise RuntimeError(
        "the source tree changed after this job started: "
        f"{SOURCE_FINGERPRINT_ENV}={expected} but the tree now hashes to "
        f"{observed}. A cluster job reads the shared checkout live, so an arm "
        "starting now would train different code than the arms before it. "
        "Restore the tree to the pinned state, or start a new job."
    )

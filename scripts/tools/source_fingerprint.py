"""Print a content hash of the source tree (no git, no torch required).

Cluster jobs read the shared checkout live.  A launcher pins the tree by
exporting this value as ``HEATMAPVLN_SOURCE_FINGERPRINT``; every arm started
afterwards refuses to run if the tree has moved (see
``scripts.training.manifest.check_pinned_source_fingerprint``).

The implementation is loaded straight from its module file rather than through
``scripts.training``: importing the package pulls in torch, and this tool has to
run in the blank container's preflight before anything heavy is available.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]
_SPEC = importlib.util.spec_from_file_location(
    "_heatmapvln_source_fingerprint", _REPO / "scripts" / "training" / "source_fingerprint.py"
)
_MANIFEST = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MANIFEST)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("project_dir", nargs="?", default=str(_REPO))
    parser.add_argument("--json", action="store_true", help="print the full record")
    args = parser.parse_args()
    record = _MANIFEST.compute_source_fingerprint(Path(args.project_dir))
    print(json.dumps(record, sort_keys=True) if args.json else record["fingerprint"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

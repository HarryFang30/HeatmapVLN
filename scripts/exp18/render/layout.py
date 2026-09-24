"""Where the EXP-18 tier C/D/E renders, configs and logs live.

Everything is derived from ``scripts/exp18/common.py`` (so ``EXP18_RENDER_ROOT``
and ``EXP18_ROOT`` overrides apply).  For a tier whose clips live in
``<RENDER_ROOT>/<tier_dir>/raw/<split>`` (``common.TIERS[t]["data_root"]``):

    <RENDER_ROOT>/configs/{C,D,E}.yaml ...      collector configs + selection manifest
    <RENDER_ROOT>/<tier_dir>/raw                collector --output (lock, stats, marker)
    <RENDER_ROOT>/<tier_dir>/raw/<split>        tier data root (AMB3R builder input)
    <RENDER_ROOT>/<tier_dir>/raw/excluded_short clips with < 20 frames, moved by finalize
    <RENDER_ROOT>/<tier_dir>/logs               launcher, worker and Xvfb logs

``python -m scripts.exp18.render.layout --shell TIER`` prints these as shell
assignments for ``run_render.sh``; ``--worker-config`` writes a per-worker
copy of a tier config restricted to a round-robin share of its scenes;
``--missing CONFIG RAW`` prints the config's selected episode ids that have no
clip (meta.json) under ``RAW/<split>`` or ``RAW/excluded_*``.

Standard library only, Python 3.8 compatible (runs under envs/vlnce).
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import sys
from pathlib import Path

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from scripts.exp18 import common  # noqa: E402

RENDER_TIERS = ("C", "D", "E")
CONFIG_DIR = Path(os.environ.get("EXP18_RENDER_CONFIG_DIR", str(common.RENDER_ROOT / "configs")))
MANIFEST_NAME = "selection_manifest.json"

# Launcher defaults.  C/D keep the v2 collector budget (300 steps); designed
# routes are up to ~2x an R2R path, so E gets a larger budget (its yaml raises
# ENVIRONMENT.MAX_EPISODE_STEPS to match).
BASE_DISPLAY = {"C": 310, "D": 330, "E": 350}
MAX_STEPS = {"C": 300, "D": 300, "E": 800}
E_MAX_EPISODE_STEPS = 1000
# HM3D scenes are big and every episode-sharded worker would load all 30 of
# them; shard D by scene instead.  C/E have 11 scenes, episode sharding balances better.
SHARD_BY = {"C": "episode", "D": "scene", "E": "episode"}

MIN_FRAMES = common.MIN_FRAMES_FOR_AMB3R


def _check(tier: str) -> str:
    if tier not in RENDER_TIERS:
        raise ValueError(f"tier must be one of {RENDER_TIERS}, got {tier!r}")
    return tier


def data_root(tier: str) -> Path:
    return Path(common.TIERS[_check(tier)]["data_root"])


def collector_output(tier: str) -> Path:
    return data_root(tier).parent


def collector_split(tier: str) -> str:
    return data_root(tier).name


def tier_dir(tier: str) -> Path:
    return collector_output(tier).parent


def log_dir(tier: str) -> Path:
    return tier_dir(tier) / "logs"


def excluded_dir(tier: str, kind: str) -> Path:
    """Sibling of the data root, so the AMB3R plan builder never scans it."""
    return collector_output(tier) / f"excluded_{kind}"


def incomplete_dir(tier: str) -> Path:
    return collector_output(tier) / "_incomplete"


def finalized_marker(tier: str) -> Path:
    return collector_output(tier) / ".exp18_finalized.json"


def manifest_path(config_dir: Path | None = None) -> Path:
    return Path(config_dir or CONFIG_DIR) / MANIFEST_NAME


def config_path(tier: str, config_dir: Path | None = None) -> Path:
    return Path(config_dir or CONFIG_DIR) / f"{_check(tier)}.yaml"


def finalize_report_path(tier: str) -> Path:
    return common.clip_list_path(tier).with_name(f"{tier}_finalize_report.json")


def shell_assignments(tier: str) -> str:
    values = {
        "DATA_ROOT": data_root(tier),
        "OUTPUT": collector_output(tier),
        "SPLIT": collector_split(tier),
        "TIER_DIR": tier_dir(tier),
        "DEFAULT_CONFIG": config_path(tier),
        "DEFAULT_LOG_DIR": log_dir(tier),
        "DEFAULT_BASE_DISPLAY": BASE_DISPLAY[tier],
        "DEFAULT_MAX_STEPS": MAX_STEPS[tier],
        "DEFAULT_SHARD_BY": SHARD_BY[tier],
        "FINALIZED_MARKER": finalized_marker(tier),
        "INCOMPLETE_DIR": incomplete_dir(tier),
        "MANIFEST": manifest_path(),
    }
    return "\n".join(f"{key}={shlex.quote(str(value))}" for key, value in values.items())


def _content_scenes_line(lines):
    hits = [i for i, line in enumerate(lines) if line.strip().startswith("CONTENT_SCENES:")]
    if len(hits) != 1:
        raise ValueError(f"expected exactly one CONTENT_SCENES line, found {len(hits)}")
    return hits[0]


def write_worker_configs(config: Path, num_workers: int, out_dir: Path) -> list:
    """Split a tier config's CONTENT_SCENES round-robin (sorted) over workers.

    Returns the written paths; workers that would get no scene are omitted.
    select_episodes.py writes CONTENT_SCENES as a one-line JSON list, which is
    what makes this text edit safe.
    """
    lines = Path(config).read_text().splitlines()
    index = _content_scenes_line(lines)
    prefix, _, payload = lines[index].partition("CONTENT_SCENES:")
    scenes = sorted(json.loads(payload))
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for worker in range(min(num_workers, len(scenes))):
        share = scenes[worker::num_workers]
        worker_lines = list(lines)
        worker_lines[index] = f"{prefix}CONTENT_SCENES: {json.dumps(share)}"
        target = out_dir / f"{Path(config).stem}_w{worker}.yaml"
        target.write_text("\n".join(worker_lines) + "\n")
        written.append(target)
    return written


def _config_list(lines, key: str) -> list:
    hits = [line for line in lines if line.strip().startswith(f"{key}:")]
    if len(hits) != 1:
        raise ValueError(f"expected exactly one {key} line, found {len(hits)}")
    return json.loads(hits[0].split(":", 1)[1])


def missing_episodes(config: Path, raw: Path) -> list:
    """Selected episode ids (EPISODES_ALLOWED, one-line JSON) without any clip.

    A clip counts wherever finalize looks for it: the data root and the
    excluded_* siblings (a short clip moved aside was rendered).  Episode ids
    are unique within every EXP-18 tier config, like EPISODES_ALLOWED assumes.
    """
    allowed = [str(e) for e in _config_list(Path(config).read_text().splitlines(), "EPISODES_ALLOWED")]
    rendered = set()
    for meta in list(Path(raw).glob("*/*/clip_*/meta.json")):
        if meta.parts[-4].startswith("_"):  # _incomplete/<stamp>/... is not a clip
            continue
        try:
            rendered.add(str(json.loads(meta.read_text()).get("episode_id")))
        except (OSError, ValueError):
            continue
    return [e for e in allowed if e not in rendered]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--shell", metavar="TIER", help="print shell assignments for run_render.sh")
    group.add_argument("--worker-config", nargs=3, metavar=("CONFIG", "NUM_WORKERS", "OUT_DIR"),
                       help="write per-worker scene-sharded configs; prints one path per line")
    group.add_argument("--missing", nargs=2, metavar=("CONFIG", "RAW"),
                       help="print selected episode ids of CONFIG with no clip under RAW (one per line)")
    args = parser.parse_args()
    if args.shell:
        print(shell_assignments(args.shell))
        return
    if args.missing:
        for episode_id in missing_episodes(Path(args.missing[0]), Path(args.missing[1])):
            print(episode_id)
        return
    config, num_workers, out_dir = args.worker_config
    for path in write_worker_configs(Path(config), int(num_workers), Path(out_dir)):
        print(path)


if __name__ == "__main__":
    main()

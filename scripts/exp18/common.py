"""Shared constants for EXP-18 (first-person topology heatmap visualization).

Every EXP-18 tool imports its paths and tier definitions from here, so the
pre-registered tiers (docs/experiments/README.md, EXP-18) live in one place.

This module must stay importable from BOTH Python environments on the dev
machine: ``envs/vlnce`` (Python 3.8, habitat-sim, no torch/matplotlib) and
``envs/qwen25`` (Python 3.12, torch, matplotlib).  Keep it standard-library
only and Python 3.8 compatible.
"""
from __future__ import annotations

import hashlib
import os
from pathlib import Path

WORKSPACE = Path(os.environ.get("EXP18_WORKSPACE", "/mnt/afs/liwenhao/agent/370910109"))

# Outputs (dumps, metrics, figures, logs, staged source) and new renders.
EXP_ROOT = Path(os.environ.get("EXP18_ROOT", str(WORKSPACE / "model" / "exp18_first_person_viz")))
RENDER_ROOT = Path(os.environ.get("EXP18_RENDER_ROOT", str(WORKSPACE / "data" / "exp18_renders")))

QWEN_PYTHON = WORKSPACE / "envs" / "qwen25" / "bin" / "python"
VLNCE_PYTHON = WORKSPACE / "envs" / "vlnce" / "bin" / "python"
VLNCE_PROJECT = WORKSPACE / "habitat" / "VLN-CE"
X11_BUNDLE = WORKSPACE / "tools" / "x11_headless_bundle_ubuntu22_20260801_v4"
MP3D_SCENES = VLNCE_PROJECT / "data" / "scene_datasets" / "mp3d"
HM3D_SCENES = VLNCE_PROJECT / "data" / "scene_datasets" / "hm3d" / "train"
R2R_VAL_UNSEEN_EPISODES = (
    VLNCE_PROJECT / "data" / "datasets" / "R2R_VLNCE_v1-3_preprocessed" / "val_unseen" / "val_unseen.json.gz"
)
SCALEVLN_EPISODES = VLNCE_PROJECT / "data" / "datasets" / "ScaleVLN" / "scalevln_subset_150k.json.gz"

HEAD_CHECKPOINT = WORKSPACE / "model" / "exp03_deployment_head" / "head_v2_best.pth"
HEAD_CONFIG_RELPATH = "configs/train_heatmap_internnav_single_view_8gpu.yaml"
INTERNNAV_MODEL_PATH = WORKSPACE / "InternNav-Model"
AMB3R_ROOT = WORKSPACE / "amb3r"
DA3_CHECKPOINT = AMB3R_ROOT / "checkpoints" / "DA3NESTED-GIANT-LARGE"

R2R_V2_ROOT = WORKSPACE / "r2r_panoramic_data_v2" / "train"
R2R_V2_AMB3R_CACHE = WORKSPACE / "data" / "amb3r_endpoint_v3_full_r2r"

# The four R2R v2 scenes the head never trained on (MD5 split; also held out in
# the random-walk pre-training its lineage started from).
R2R_V2_VAL_SCENES = ("JeFG25nYj2p", "JmbYfDe2QKZ", "ZMojNkEp431", "b8cTxDM8gDG")

# Dev-machine rule (ledger §0.7): at most three GPUs, never GPU 0.
GPU_IDS = (7, 6, 5)

NUM_HISTORY = 8
MIN_HISTORY = 5
MIN_FRAMES_FOR_AMB3R = 20  # AMB3R map init window

TIERS = {
    "A": {
        "name": "train_scenes",
        "label": "Training scenes (R2R)",
        "data_root": R2R_V2_ROOT,
        "cache_root": R2R_V2_AMB3R_CACHE,
        "clips_per_scene": 10,
    },
    "B": {
        "name": "heldout_scenes",
        "label": "Held-out scenes (R2R)",
        "data_root": R2R_V2_ROOT,
        "cache_root": R2R_V2_AMB3R_CACHE,
        "clips_per_scene": None,  # all clips
    },
    "C": {
        "name": "val_unseen",
        "label": "Unseen scenes (R2R val-unseen)",
        "data_root": RENDER_ROOT / "C_val_unseen" / "raw" / "val_unseen",
        "cache_root": RENDER_ROOT / "amb3r_cache" / "C",
        "episodes_rendered_per_scene": 30,
        "clips_per_scene": 25,
    },
    "D": {
        "name": "hm3d",
        "label": "Cross-dataset (HM3D)",
        "data_root": RENDER_ROOT / "D_hm3d" / "raw" / "train",
        "cache_root": RENDER_ROOT / "amb3r_cache" / "D",
        "num_scenes": 30,
        "clips_per_scene": 4,
    },
    "E": {
        "name": "designed_routes",
        "label": "Designed routes (out-and-back, loop)",
        "data_root": RENDER_ROOT / "E_designed" / "raw" / "val_unseen",
        "cache_root": RENDER_ROOT / "amb3r_cache" / "E",
        "out_and_back_per_scene": 2,
        "loop_per_scene": 2,
    },
}


def sha1_key(text: str) -> str:
    """Deterministic ordering key used by every pre-registered selection rule."""
    return hashlib.sha1(text.encode("utf-8")).hexdigest()


def clip_list_path(tier: str) -> Path:
    return EXP_ROOT / "clip_lists" / f"{tier}.txt"


def read_clip_list(tier: str, path: Path | None = None) -> list:
    """Return ``<scene>/<clip>`` keys (relative to the tier's data root)."""
    source = Path(path) if path is not None else clip_list_path(tier)
    keys = []
    for line in source.read_text().splitlines():
        line = line.strip()
        if line and not line.startswith("#"):
            keys.append(line)
    return keys


def write_clip_list(tier: str, keys, header: str = "", path: Path | None = None) -> Path:
    target = Path(path) if path is not None else clip_list_path(tier)
    target.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"# {row}" for row in header.splitlines() if row.strip()]
    lines.extend(keys)
    target.write_text("\n".join(lines) + "\n")
    return target

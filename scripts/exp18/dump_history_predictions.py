#!/usr/bin/env python3
"""EXP-18: dump the deployed History Head's predictions under VO and GT poses.

For every clip of one tier's pre-registered clip list
(``common.read_clip_list(tier)``, keys ``<scene>/<clip>`` under
``common.TIERS[tier]['data_root']``) this writes one compressed npz with, per
query row, the GT four-view history labels and the frozen deployed head's
eval-mode outputs under two pose arms that share *identical* visual features:

* ``vo``: AMB3R causal endpoint-cache relative poses (deployed setting, the
  arm every EXP-18 criterion is judged on);
* ``gt``: Habitat simulator relative poses, computed by the dataset itself
  (``compute_history_rel_poses(..., camera_forward_axis='-z')``).

Population (pre-registered): exactly the cache endpoint rows
t = 19, 27, 35, ... plus the final frame; history = ``linspace(0, t-1, 8)``.
Both arms run on the same rows.  Head = ``common.HEAD_CHECKPOINT``
(``initial_head_hash`` checked on load), built like EXP-03
(``scripts/tools/diagnose_heatmap_shortcuts.py``: diagnostic ``load_config`` +
``build_model`` + ``load_heatmap_head_checkpoint`` +
``AutoProcessor(use_fast=False)``), bf16 autocast, batch size 1.

How the arms are fed:

* The dataset runs in **GT-pose mode** over the explicit clip list; its RGB and
  labels are byte-identical to cache mode.  VO poses are read with the same
  strict reader the dataset uses in cache mode (``AMB3RPoseCache.lookup``),
  which fails closed unless the cache's history IDs equal the dataset's
  ``_sample_history_indices(0, t, K)``.  An absent/invalid cache is an error
  (``--missing-cache error``, default), never replaced by GT poses.
* One frozen visual forward per row, then the eval-mode head twice (vo, gt) on
  the same detached features.  The first row is self-checked against
  ``model.forward``; the tolerance is the measured MACA bf16 noise floor
  (repeat forwards differ by ~1e-4 gated, ~0.125 visibility logits).

npz schema (``<out>/<scene>/<clip>.npz``; N rows, K=8 history slots padded,
T clip frames, H=W=64; view axis order is ``direction_order`` = F,R,B,L):

  identity  schema, tier, scene, clip, clip_key ("<scene>/<clip>"), clip_dir,
            scene_id, episode_id (from meta.json, "" if absent), meta_json
            (the clip's full meta.json), direction_order, pose_convention,
            arms (["vo","gt"])
  clip      frame_count (T), clip_c2w [T,4,4] front-camera c2w of every frame
            (Habitat world, y up, camera looks along -Z),
            cache_endpoint_frame_ids [E]
  per row   current_frame_ids [N] int64, is_final_frame [N], vo_available [N]
            history_frame_ids [N,K] int64 (-1 pad), history_mask [N,K] bool
            gt_heatmap [N,K,4,H,W] f16 (training labels), gt_visibility
              [N,K,4] u8 (left/right/back are FOV-only, no occlusion test)
            gt_view_class [N,K] int8: validate.py's 5-way target, computed on
              the f32 labels (0 = none, 1..4 = F,R,B,L; -1 pad)
            gt_view_peak_yx [N,K,4,2] int16: per-view argmax (y,x) of the f32
              labels (-1 pad)
            gt_rel_poses, vo_rel_poses [N,K,4] f32 (forward_m, left_m,
              cos_yaw, sin_yaw) of each history camera in the current front
              camera frame (NaN pad)
            current_c2w [N,4,4], current_c2w_views [N,4,4,4] (F,R,B,L
              cameras), history_c2w [N,K,4,4] (NaN pad); a history frame's
              world position is history_c2w[..., :3, 3]
  per arm   pred_<arm>_heatmaps_gated [N,K,4,H,W] f16: spatial softmax x
              softmax([0, vis_F, vis_R, vis_B, vis_L])[1:], so
              sum(gated) + none_probability = 1 per slot
            pred_<arm>_none_probability [N,K] f32
            pred_<arm>_visibility_logits [N,K,4] f32
            pred_<arm>_view_peak_yx [N,K,4,2] int16: per-view argmax of the
              head's ``heatmaps`` output (what validate.py's
              _HeatmapJointMetricAccumulator scores; exact, not from f16)
            pred_<arm>_gated_argmax [N,K,3] int16: (view, y, x) argmax of the
              f32 ``heatmaps_gated`` over 4xHxW (the bearing metric's peak)
            pred_<arm>_peak_err_px [N,K,4] f32: |view_peak - gt_view_peak| on
              GT-eligible views, NaN elsewhere (case browsing only; ignores
              the view classification, unlike joint PCK)

Sidecar ``<scene>/<clip>.json`` (commit marker: npz bytes + sha256, run
fingerprint, per-arm joint PCK@8 of the clip); shard manifests under
``manifests/``; ``manifest.json`` merged at the end (or ``--merge-manifests``),
listing clips of the tier list that are still missing.

Resumable and shardable by clip: a clip is done iff its sidecar carries the
current run fingerprint and the npz size matches.  Shards take
``sorted(clip_list)[i::n]``.

Examples (dev machine; production goes through scripts/exp18/run_dump.sh):
  CUDA_VISIBLE_DEVICES=7 $W/envs/qwen25/bin/python scripts/exp18/dump_history_predictions.py --tier B
  ... --tier B --shard-index 0 --num-shards 3
  ... --tier B --merge-manifests
"""

from __future__ import annotations

import argparse
import datetime as _dt
import fnmatch
import hashlib
import json
import logging
import os
import platform
import socket
import sys
import time
from pathlib import Path

import numpy as np

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.exp18 import common  # noqa: E402

SCHEMA = "heatmapvln-exp18-history-head-dump-v2"
DIRECTIONS = ("front", "right", "back", "left")
POSE_CONVENTION = "habitat_c2w_minus_z__forward_left_cos_yaw_sin_yaw__v1"
POPULATION = "amb3r_cache_endpoints"
# transformers 4.51 Qwen2.5-VL SDPA vision attention builds a dense [L,L]
# block-diagonal mask over all images of a forward, so batching is slower
# (B=4 measured 0.77 vs 1.32 rows/s on one C500).
BATCH_SIZE = 1
LOGGER = logging.getLogger("exp18_dump")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--tier", default=os.environ.get("TIER"), choices=sorted(common.TIERS),
                   help="EXP-18 tier (env TIER); sets data root, cache root, clip list and output dir")
    p.add_argument("--repo", default=str(SOURCE_ROOT), help="HeatmapVLN source tree to import (default: this one)")
    p.add_argument("--config", default=None, help=f"default: <repo>/{common.HEAD_CONFIG_RELPATH}")
    p.add_argument("--head-checkpoint", default=str(common.HEAD_CHECKPOINT))
    p.add_argument("--data-root", default=None, help="override TIERS[tier]['data_root'] (dev only)")
    p.add_argument("--cache-root", default=None, help="override TIERS[tier]['cache_root'] (dev only)")
    p.add_argument("--clip-list", default=None, help="override common.clip_list_path(tier) (dev only)")
    p.add_argument("--output-dir", default=None, help="default: EXP_ROOT/dumps/<tier>")
    p.add_argument("--missing-cache", choices=("error", "skip"), default="error",
                   help="clip without a valid AMB3R cache: abort (default) or record it as skipped/failed")
    p.add_argument("--internnav-model-path",
                   default=os.environ.get("INTERNNAV_MODEL_PATH", str(common.INTERNNAV_MODEL_PATH)))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--amp", choices=("bf16", "none"), default="bf16",
                   help="bf16 matches training (optim.amp) and EXP-03's evaluation")
    p.add_argument("--num-workers", type=int, default=int(os.environ.get("DUMP_NUM_WORKERS", 6)))
    p.add_argument("--clips", default="", help="comma-separated <scene>/<clip> fnmatch patterns (dev filter)")
    p.add_argument("--max-clips", type=int, default=0, help="first N clips of the sorted list (dev only)")
    p.add_argument("--shard-index", type=int, default=int(os.environ.get("EXP18_SHARD_INDEX", 0)))
    p.add_argument("--num-shards", type=int, default=int(os.environ.get("EXP18_NUM_SHARDS", 1)))
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--no-self-check", action="store_true",
                   help="skip the split-forward vs model.forward check on the first row")
    p.add_argument("--merge-manifests", action="store_true",
                   help="only merge shard manifests + per-clip sidecars into <output-dir>/manifest.json")
    p.add_argument("--dry-list", action="store_true",
                   help="plan this shard's clips and rows, then exit (no model load)")
    args = p.parse_args(argv)
    if args.tier not in common.TIERS:
        p.error(f"--tier (or env TIER) must be one of {sorted(common.TIERS)}, got {args.tier!r}")
    if not (0 <= args.shard_index < args.num_shards):
        p.error("need 0 <= --shard-index < --num-shards")
    tier = common.TIERS[args.tier]
    args.data_root = Path(args.data_root or tier["data_root"]).expanduser()
    args.cache_root = Path(args.cache_root or tier["cache_root"]).expanduser()
    args.clip_list = Path(args.clip_list or common.clip_list_path(args.tier)).expanduser()
    args.output_dir = Path(args.output_dir or common.EXP_ROOT / "dumps" / args.tier).expanduser()
    return args


# --------------------------------------------------------------------------- #
# Small utilities
# --------------------------------------------------------------------------- #
def sha256_file(path: str | Path, chunk: int = 1 << 22) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


def utc_now() -> str:
    return _dt.datetime.now(_dt.timezone.utc).isoformat()


def write_json_atomic(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=True), encoding="utf-8")
    os.replace(tmp, path)


def write_npz_atomic(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    with open(tmp, "wb") as handle:  # a handle stops numpy from appending ".npz"
        np.savez_compressed(handle, **arrays)
    os.replace(tmp, path)


def resolve_git_sha(repo: Path) -> str:
    """EXP18_GIT_SHA, else <repo>/.exp18_git_sha (staged archives have no .git)."""
    sha = os.environ.get("EXP18_GIT_SHA", "").strip()
    if sha:
        return sha
    marker = repo / ".exp18_git_sha"
    if marker.is_file():
        return marker.read_text(encoding="utf-8").strip()
    LOGGER.warning("no EXP18_GIT_SHA and no %s: code_git_sha=unknown", marker)
    return "unknown"


def peak_yx(maps: np.ndarray) -> np.ndarray:
    """Argmax (y, x) over the last two axes (first occurrence, like torch)."""
    flat = maps.reshape(*maps.shape[:-2], -1).argmax(-1)
    return np.stack([flat // maps.shape[-1], flat % maps.shape[-1]], axis=-1)


def gt_view_class(gt_hm: np.ndarray, gt_vis: np.ndarray) -> np.ndarray:
    """validate.py's 5-way target: argmax peak over eligible views (+1), 0 = none."""
    peak = gt_hm.reshape(*gt_hm.shape[:-2], -1).max(-1)  # [k,4]
    eligible = (gt_vis > 0.5) & (peak > 0)
    view = np.where(eligible, peak, -np.inf).argmax(-1)
    return np.where(eligible.any(-1), view + 1, 0).astype(np.int8)


def joint_pck(gt_class: np.ndarray, gt_peaks: np.ndarray, vis_logits: np.ndarray,
              pred_peaks: np.ndarray, valid: np.ndarray, radius: float = 8.0) -> dict:
    """Numpy mirror of _HeatmapJointMetricAccumulator on valid slots (for sidecars/smoke)."""
    gt_class, gt_peaks = gt_class[valid], gt_peaks[valid]
    vis_logits, pred_peaks = vis_logits[valid], pred_peaks[valid]
    pred_class = np.concatenate([np.zeros_like(vis_logits[:, :1]), vis_logits], axis=-1).argmax(-1)
    visible = gt_class > 0
    n = int(visible.sum())
    if n == 0:
        return {"visible_slots": 0, "joint_pck8": None, "view5_acc": None}
    rows = np.nonzero(visible)[0]
    target = gt_class[rows] - 1
    d2 = ((pred_peaks[rows, target].astype(np.int64) - gt_peaks[rows, target].astype(np.int64)) ** 2).sum(-1)
    correct = pred_class[rows] == gt_class[rows]
    return {"visible_slots": n,
            "joint_pck8": float((correct & (d2 <= radius ** 2)).mean()),
            "view5_acc": float((pred_class == gt_class).mean())}


# --------------------------------------------------------------------------- #
# Clip plan
# --------------------------------------------------------------------------- #
def planned_clips(args: argparse.Namespace) -> list[str]:
    keys = sorted(set(common.read_clip_list(args.tier, args.clip_list)))
    patterns = [c for c in args.clips.split(",") if c]
    if patterns:
        keys = [k for k in keys if any(fnmatch.fnmatch(k, pat) for pat in patterns)]
    if args.max_clips > 0:
        keys = keys[: args.max_clips]
    return keys


def cache_sidecar(cache_root: Path, key: str) -> Path:
    return cache_root / key / "amb3r_pose_cache.npz.json"


# --------------------------------------------------------------------------- #
# Model / head
# --------------------------------------------------------------------------- #
def diagnostic_config(args: argparse.Namespace) -> dict:
    from scripts.tools.diagnose_heatmap_shortcuts import load_config as diagnostic_load_config

    return diagnostic_load_config(argparse.Namespace(
        config=args.config, data_root=str(args.data_root), architecture="internnav_single_view",
        device=args.device, num_history=common.NUM_HISTORY, internnav_model_path=args.internnav_model_path,
        amb3r_pose_cache_root=None,  # the dump reads VO itself, see module docstring
        amb3r_pose_cache_max_clips=16))


def load_everything(args: argparse.Namespace):
    """Build the deployed stack exactly like EXP-03 did.  Returns (cfg, model, collator, audit)."""
    import torch

    from scripts.tools.diagnose_heatmap_shortcuts import (
        build_single_view_collator,
        heatmap_head_state_dict,
        load_heatmap_head_checkpoint,
        state_hash,
    )
    from scripts.training import build_model

    cfg = diagnostic_config(args)
    heatmap_cfg = cfg["model"]["heatmap"]
    if heatmap_cfg.get("input_mode") != "internnav_single_view":
        raise RuntimeError(f"not a single-view config: {heatmap_cfg.get('input_mode')!r}")
    if heatmap_cfg.get("history_pose_convention") != POSE_CONVENTION:
        raise RuntimeError(f"unexpected pose convention {heatmap_cfg.get('history_pose_convention')!r}")

    t0 = time.time()
    model = build_model(cfg, verbose=False, device=args.device, enable_action_head=False)
    model.qwen2_5_vl._load_model()
    model._ensure_heatmap_vln()
    t_model = time.time() - t0

    initial_hash, payload = load_heatmap_head_checkpoint(model.heatmap_vln, args.head_checkpoint)
    loaded_hash = state_hash(heatmap_head_state_dict(model.heatmap_vln))
    if loaded_hash != initial_hash:
        raise RuntimeError(f"head state hash {loaded_hash} != checkpoint initial_head_hash {initial_hash}")
    for param in model.parameters():
        param.requires_grad_(False)
    model.eval()
    head = model.heatmap_vln
    if not getattr(head, "joint_panorama_inference", False):
        raise RuntimeError("head.joint_panorama_inference is False: none_probability would be absent")
    if tuple(getattr(head, "output_direction_order", DIRECTIONS)) != DIRECTIONS:
        raise RuntimeError(f"unexpected head view order {getattr(head, 'output_direction_order', None)}")
    if model.single_view_heatmap_extractor is None:
        raise RuntimeError("single-view extractor was not constructed")

    t1 = time.time()
    collator = build_single_view_collator(cfg)  # AutoProcessor(use_fast=False), as in training
    audit = {
        "head_checkpoint": str(Path(args.head_checkpoint).resolve()),
        "initial_head_hash": initial_hash,
        "loaded_state_hash_matches": True,
        "provenance": payload.get("provenance"),
        "architecture_id": getattr(head, "architecture_id", None),
        "output_direction_order": list(DIRECTIONS),
        "joint_panorama_inference": True,
        "visual_dtype": str(next(model.single_view_heatmap_extractor._visual.parameters()).dtype),
        "head_param_dtype": str(next(head.parameters()).dtype),
        "processor_use_fast": False,
        "timing_s": {"build_and_load_backbone": round(t_model, 1), "processor": round(time.time() - t1, 1)},
        "torch": torch.__version__,
    }
    return cfg, model, collator, audit


def features_to_decoder(features, decoder_device, history_mask):
    """Same cast as pipeline._forward_frozen_single_view_heatmap."""
    import torch

    return type(features)(
        current_vit={k: v.to(device=decoder_device, dtype=torch.float32) for k, v in features.current_vit.items()},
        current_merged=features.current_merged.to(device=decoder_device, dtype=torch.float32),
        history_vit={k: v.to(device=decoder_device, dtype=torch.float32) for k, v in features.history_vit.items()},
        history_merged=features.history_merged.to(device=decoder_device, dtype=torch.float32),
        history_queries=features.history_queries.to(device=decoder_device, dtype=torch.float32),
        history_mask=history_mask.to(device=decoder_device, dtype=torch.bool),
    )


def forward_arms(model, batch: dict, arm_poses: dict) -> dict:
    """One frozen visual forward, then the eval-mode head once per pose arm."""
    extractor = model.single_view_heatmap_extractor
    head = model.heatmap_vln
    visual = extractor._visual
    visual.eval()
    head.eval()
    visual_device = next(visual.parameters()).device
    decoder_device = next(head.parameters()).device
    features = extractor.extract_from_pixels(
        pixel_values=batch["pixel_values"].to(visual_device, non_blocking=True),
        image_grid_thw=batch["image_grid_thw"].to(visual_device, non_blocking=True),
        num_histories=batch["num_histories"],
    )
    explicit = batch["history_mask"]
    if tuple(explicit.shape) != tuple(features.history_mask.shape):
        raise ValueError("collator and extractor history masks disagree")
    features = features_to_decoder(features, decoder_device, explicit)
    return {arm: head(features, poses.to(device=decoder_device, dtype=features.history_queries.dtype))
            for arm, poses in arm_poses.items()}


# --------------------------------------------------------------------------- #
# Dataset plumbing
# --------------------------------------------------------------------------- #
def make_dataset(clip_dirs: list[Path], data_root: Path, cfg: dict):
    from src.data.sliding_window_dataset import VLNSlidingWindowDataset

    class _ClipListDataset(VLNSlidingWindowDataset):
        """Index exactly the given clips; everything else is the stock dataset."""

        def _enumerate_clips(self):  # noqa: D401
            if not clip_dirs:
                raise FileNotFoundError("empty clip list")
            return list(clip_dirs)

    sw = cfg["data"]["sliding_window"]
    return _ClipListDataset(
        root=str(data_root),
        split="all",
        min_history=int(sw["min_history"]),                # 5
        num_history_sample=int(sw["num_history_sample"]),  # 8
        image_size=tuple(cfg["data"]["image_size"]),       # (384, 384)
        hm_size=tuple(cfg["data"]["init_hm_size"]),        # (64, 64)
        load_depth=True,
        cache_poses=True,
        sample_stride=1,              # not the config's 2: that would also drop endpoint rows
        enable_augmentation=False,
        clip_level_sampling=False,    # deterministic sliding-window index
        samples_per_clip=1,
        defer_heatmap_to_gpu=False,   # labels computed on CPU in the worker
        load_single_view_history_frames=True,
        single_view_rgb_input=True,   # front-only RGB into the model
        amb3r_pose_cache_root=None,   # GT-pose mode; VO read separately with the same strict reader
        require_amb3r_pose_cache=False,
        max_clips=0,
    )


class DumpItems:
    """torch Dataset over planned (dataset_index, clip_idx, t) rows."""

    def __init__(self, base, items, pose_cache):
        self.base = base
        self.items = items
        self.pose_cache = pose_cache

    def __len__(self):
        return len(self.items)

    def __getitem__(self, i):
        from src.data.trajectory_utils import compute_history_rel_poses

        base = self.base
        ds_index, clip_idx, t = self.items[i]
        if tuple(int(v) for v in base.sample_index[ds_index]) != (clip_idx, t):
            raise RuntimeError("sample_index drifted")
        failures = int(base._sample_failure_count)
        sample = base[ds_index]
        if int(base._sample_failure_count) != failures:
            raise RuntimeError(f"dataset returned a dummy sample for {base.clips[clip_idx]}@{t}")
        if sample.get("history_pose_provider") != "habitat_gt":
            raise RuntimeError(f"unexpected provider {sample.get('history_pose_provider')!r}")
        clip_dir = base.clips[clip_idx]
        hist = base._sample_history_indices(0, t, base.num_history_sample).astype(np.int64)
        poses = base._load_poses(clip_idx)  # list[T] of front c2w [4,4]
        hist_c2w = np.stack([poses[j] for j in hist]).astype(np.float32)
        cur_c2w = np.asarray(poses[t], dtype=np.float32)
        gt_rel = np.asarray(sample["history_rel_poses"], dtype=np.float32)
        recomputed = compute_history_rel_poses(list(hist_c2w), cur_c2w, camera_forward_axis="-z")
        if not np.allclose(recomputed, gt_rel, atol=1e-5):
            raise RuntimeError(f"GT rel-pose recomputation mismatch at {clip_dir}@{t}")
        views = np.full((4, 4, 4), np.nan, dtype=np.float32)
        if base._get_storage_format(clip_idx) == "chunks":
            for v, d in enumerate(DIRECTIONS):
                try:
                    views[v] = base._get_chunk_frame_array(clip_idx, t, "pose", direction=d)
                except Exception:  # noqa: BLE001 - display-only field
                    pass
        vo_rel = self.pose_cache.lookup(clip_dir, current_frame_id=t, history_frame_ids=hist)
        gt_hm = sample["heatmap"].numpy().astype(np.float32)          # [k,4,64,64]
        gt_vis = sample["gt_visibility"].numpy().astype(np.float32)   # [k,4]
        extras = {
            "clip_idx": clip_idx,
            "t": t,
            "history_frame_ids": hist,
            "gt_rel": gt_rel,
            "vo_rel": vo_rel.astype(np.float32),
            "current_c2w": cur_c2w,
            "current_c2w_views": views,
            "history_c2w": hist_c2w,
            "gt_heatmap": gt_hm,
            "gt_visibility": gt_vis,
            "gt_view_class": gt_view_class(gt_hm, gt_vis),
            "gt_view_peak_yx": peak_yx(gt_hm).astype(np.int16),
            "sample_identity": sample.get("sample_identity"),
        }
        return sample, extras


def make_collate(collator):
    def collate(items):
        return collator([s for s, _ in items]), [e for _, e in items]
    return collate


# --------------------------------------------------------------------------- #
# Per-clip buffers
# --------------------------------------------------------------------------- #
def _pad(a: np.ndarray, K: int, fill) -> np.ndarray:
    if a.shape[0] >= K:
        return a
    return np.concatenate([a, np.full((K - a.shape[0], *a.shape[1:]), fill, dtype=a.dtype)], axis=0)


class ClipBuffer:
    def __init__(self, clip_dir: Path, K: int, hm: tuple, arms: list):
        self.clip_dir, self.scene, self.clip = clip_dir, clip_dir.parent.name, clip_dir.name
        self.K, self.hm, self.arms = K, hm, arms
        self.rows: list = []
        self.t0 = time.time()

    def add(self, extras: dict, outs: dict, b: int, history_mask_row: np.ndarray) -> None:
        K, (H, W) = self.K, self.hm
        k = int(history_mask_row.sum())
        if k != len(extras["history_frame_ids"]):
            raise RuntimeError("collator history count != dataset history count")
        row = {
            "t": extras["t"],
            "history_frame_ids": _pad(extras["history_frame_ids"], K, -1),
            "history_mask": _pad(np.ones(k, dtype=bool), K, False),
            "gt_heatmap": _pad(extras["gt_heatmap"].astype(np.float16), K, 0),
            "gt_visibility": _pad(extras["gt_visibility"].astype(np.uint8), K, 0),
            "gt_view_class": _pad(extras["gt_view_class"], K, -1),
            "gt_view_peak_yx": _pad(extras["gt_view_peak_yx"], K, -1),
            "gt_rel": _pad(extras["gt_rel"], K, np.nan),
            "vo_rel": _pad(extras["vo_rel"], K, np.nan),
            "current_c2w": extras["current_c2w"],
            "current_c2w_views": extras["current_c2w_views"],
            "history_c2w": _pad(extras["history_c2w"], K, np.nan),
        }
        for arm in self.arms:
            out = outs[arm]
            gated = out["heatmaps_gated"][b, :k].float().cpu().numpy()        # f32 [k,4,H,W]
            flat = gated.reshape(k, -1).argmax(-1)
            gated_argmax = np.stack([flat // (H * W), (flat % (H * W)) // W, flat % W], axis=-1)
            view_peaks = peak_yx(out["heatmaps"][b, :k].float().cpu().numpy())  # validate.py's argmax
            row[f"pred_{arm}_heatmaps_gated"] = _pad(gated.astype(np.float16), K, np.nan)
            row[f"pred_{arm}_none_probability"] = _pad(
                out["none_probability"][b, :k].float().cpu().numpy().astype(np.float32), K, np.nan)
            row[f"pred_{arm}_visibility_logits"] = _pad(
                out["visibility"][b, :k].float().cpu().numpy().astype(np.float32), K, np.nan)
            row[f"pred_{arm}_view_peak_yx"] = _pad(view_peaks.astype(np.int16), K, -1)
            row[f"pred_{arm}_gated_argmax"] = _pad(gated_argmax.astype(np.int16), K, -1)
        self.rows.append(row)

    def arrays(self, tier: str, meta: dict, clip_c2w: np.ndarray, endpoints: np.ndarray) -> dict:
        def stack(key):
            return np.stack([r[key] for r in self.rows], axis=0)

        frame_count = int(meta["num_frames"])
        t = np.asarray([r["t"] for r in self.rows], dtype=np.int64)
        out = {
            "schema": np.asarray(SCHEMA),
            "tier": np.asarray(tier),
            "scene": np.asarray(self.scene),
            "clip": np.asarray(self.clip),
            "clip_key": np.asarray(f"{self.scene}/{self.clip}"),
            "clip_dir": np.asarray(str(self.clip_dir)),
            "scene_id": np.asarray(str(meta.get("scene_id", ""))),
            "episode_id": np.asarray(str(meta.get("episode_id", ""))),
            "meta_json": np.asarray(json.dumps(meta, ensure_ascii=False)),
            "direction_order": np.asarray(DIRECTIONS),
            "pose_convention": np.asarray(POSE_CONVENTION),
            "arms": np.asarray(self.arms),
            "frame_count": np.asarray(frame_count, dtype=np.int64),
            "clip_c2w": clip_c2w.astype(np.float32),
            "cache_endpoint_frame_ids": endpoints.astype(np.int64),
            "current_frame_ids": t,
            "is_final_frame": t == frame_count - 1,
            "vo_available": np.ones(len(t), dtype=bool),
            "history_frame_ids": stack("history_frame_ids"),
            "history_mask": stack("history_mask"),
            "gt_heatmap": stack("gt_heatmap"),
            "gt_visibility": stack("gt_visibility"),
            "gt_view_class": stack("gt_view_class"),
            "gt_view_peak_yx": stack("gt_view_peak_yx"),
            "gt_rel_poses": stack("gt_rel"),
            "vo_rel_poses": stack("vo_rel"),
            "current_c2w": stack("current_c2w"),
            "current_c2w_views": stack("current_c2w_views"),
            "history_c2w": stack("history_c2w"),
        }
        # validate.py's eligible views: GT-visible with a non-zero label peak.
        gt_peak = out["gt_heatmap"].astype(np.float32).max(axis=(-2, -1))  # [N,K,4]
        eligible = (out["gt_visibility"] > 0) & (gt_peak > 0) & out["history_mask"][..., None]
        for arm in self.arms:
            for suffix in ("heatmaps_gated", "none_probability", "visibility_logits", "view_peak_yx", "gated_argmax"):
                out[f"pred_{arm}_{suffix}"] = stack(f"pred_{arm}_{suffix}")
            diff = out[f"pred_{arm}_view_peak_yx"].astype(np.float32) - out["gt_view_peak_yx"].astype(np.float32)
            err = np.linalg.norm(diff, axis=-1)
            err[~eligible] = np.nan
            out[f"pred_{arm}_peak_err_px"] = err.astype(np.float32)
        return out

    def summary(self, arrays: dict) -> dict:
        valid = arrays["history_mask"]
        res = {"rows": len(self.rows), "valid_slots": int(valid.sum()),
               "gt_visible_slots": int((arrays["gt_view_class"][valid] > 0).sum())}
        for arm in self.arms:
            err = arrays[f"pred_{arm}_peak_err_px"]
            finite = np.isfinite(err)
            res[arm] = joint_pck(arrays["gt_view_class"], arrays["gt_view_peak_yx"],
                                 arrays[f"pred_{arm}_visibility_logits"], arrays[f"pred_{arm}_view_peak_yx"], valid)
            res[arm]["median_view_peak_err_px"] = float(np.median(err[finite])) if finite.any() else None
        return res


# --------------------------------------------------------------------------- #
# Manifest merge
# --------------------------------------------------------------------------- #
def merge_manifests(args: argparse.Namespace) -> Path:
    out_dir = args.output_dir
    shards = sorted((out_dir / "manifests").glob("shard-*.json"))
    shard_payloads = [json.loads(p.read_text(encoding="utf-8")) for p in shards]
    fingerprints = sorted({s.get("run_fingerprint") for s in shard_payloads})
    clips = []
    for sidecar in sorted(out_dir.glob("*/clip_*.json")):
        payload = json.loads(sidecar.read_text(encoding="utf-8"))
        clips.append({k: payload.get(k) for k in ("clip_key", "npz", "npz_bytes", "npz_sha256", "rows",
                                                   "run_fingerprint", "summary")})
    expected = sorted(set(common.read_clip_list(args.tier, args.clip_list))) if args.clip_list.is_file() else []
    have = {c["clip_key"] for c in clips}
    missing = [k for k in expected if k not in have]
    failed = [f for s in shard_payloads for f in s.get("clips_failed", [])]
    skipped = [f for s in shard_payloads for f in s.get("clips_skipped", [])]
    consistent = len(fingerprints) == 1 and all(c["run_fingerprint"] in fingerprints for c in clips)
    merged = {
        "schema": SCHEMA + "/manifest",
        "tier": args.tier,
        "merged_at_utc": utc_now(),
        "clip_list": str(args.clip_list),
        "clip_list_sha256": sha256_file(args.clip_list) if args.clip_list.is_file() else None,
        "run_fingerprints": fingerprints,
        "consistent": consistent,
        "complete": consistent and bool(expected) and not missing,
        "run": shard_payloads[-1].get("run") if shard_payloads else None,
        "shards": [{k: s.get(k) for k in ("shard", "started_utc", "finished_utc", "clips_planned",
                                          "clips_already_done", "timing_s")}
                   | {"clips_done": len(s.get("clips_done", []))} for s in shard_payloads],
        "totals": {"clips_expected": len(expected), "clips": len(clips), "clips_missing": len(missing),
                   "clips_failed": len(failed), "clips_skipped": len(skipped),
                   "rows": int(sum(c["rows"] or 0 for c in clips))},
        "clips_missing": missing,
        "clips_failed": failed,
        "clips_skipped": skipped,
        "clips": clips,
    }
    path = out_dir / "manifest.json"
    write_json_atomic(path, merged)
    return path


# --------------------------------------------------------------------------- #
# Main
# --------------------------------------------------------------------------- #
def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    out_dir = args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    if args.merge_manifests:
        path = merge_manifests(args)
        print(path)
        print(json.dumps(json.loads(path.read_text(encoding="utf-8"))["totals"]))
        return 0

    if args.device.startswith("cuda") and not os.environ.get("CUDA_VISIBLE_DEVICES", "").strip():
        # --device cuda:0 would otherwise silently mean physical GPU 0 (dev rule: never GPU 0).
        raise SystemExit("set CUDA_VISIBLE_DEVICES explicitly (run_dump.sh does this per shard)")
    repo = Path(args.repo).expanduser().resolve(strict=True)
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    args.config = str(Path(args.config or repo / common.HEAD_CONFIG_RELPATH).resolve(strict=True))
    data_root = args.data_root
    if not data_root.is_dir():
        raise FileNotFoundError(f"tier {args.tier} data root missing: {data_root}")
    cache_root = args.cache_root.resolve(strict=True)

    # diagnostic load_config overrides data.root/llm.model_path, so the config's
    # ${VAR} placeholders are unused on this path; export them anyway.
    os.environ.setdefault("HEATMAP_DATA_ROOT", str(data_root))
    os.environ.setdefault("SINGLE_VIEW_HM_OUT_DIR", str(out_dir))
    os.environ.setdefault("SINGLE_VIEW_HM_TB_DIR", str(out_dir / "tensorboard"))
    os.environ.setdefault("INTERNNAV_MODEL_PATH", str(args.internnav_model_path))
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if not Path(args.internnav_model_path, "preprocessor_config.json").is_file():
        raise FileNotFoundError(f"InternNav model dir looks wrong: {args.internnav_model_path}")

    # ---- clip plan (cheap, no model) --------------------------------------
    all_keys = planned_clips(args)
    shard_keys = all_keys[args.shard_index :: args.num_shards]
    LOGGER.info("tier %s: %d clips planned, %d in shard %d/%d", args.tier, len(all_keys), len(shard_keys),
                args.shard_index, args.num_shards)
    skipped: list = []
    failed: list = []
    kept: list = []
    for key in shard_keys:
        clip_dir = data_root / key
        if not (clip_dir / "meta.json").is_file():
            raise FileNotFoundError(f"clip-list entry has no meta.json: {clip_dir}")
        if cache_sidecar(cache_root, key).is_file():
            kept.append(key)
        elif args.missing_cache == "error":
            raise FileNotFoundError(f"no AMB3R cache for {key} under {cache_root}")
        else:
            skipped.append({"clip": key, "reason": "no_amb3r_cache"})
    shard_keys = kept

    ckpt_sha = sha256_file(args.head_checkpoint)
    run = {
        "schema": SCHEMA,
        "tier": args.tier,
        "tier_label": common.TIERS[args.tier]["label"],
        "head_checkpoint": str(Path(args.head_checkpoint).resolve()),
        "head_checkpoint_sha256": ckpt_sha,
        "config": args.config,
        "config_sha256": sha256_file(args.config),
        "dump_script_sha256": sha256_file(__file__),
        "code_repo": str(repo),
        "code_git_sha": resolve_git_sha(repo),
        "internnav_model_path": str(args.internnav_model_path),
        "data_root": str(data_root),
        "amb3r_cache_root": str(cache_root),
        "clip_list": str(args.clip_list),
        "clip_list_sha256": sha256_file(args.clip_list),
        "population": POPULATION,
        "batch_size": BATCH_SIZE,
        "amp": args.amp,
        "arms": ["vo", "gt"],
        "processor_use_fast": False,
    }
    fingerprint_src = {k: run[k] for k in ("schema", "tier", "head_checkpoint_sha256", "config_sha256", "population",
                                            "amp", "arms", "data_root", "amb3r_cache_root")}
    run_fingerprint = hashlib.sha256(json.dumps(fingerprint_src, sort_keys=True).encode()).hexdigest()[:16]

    def done(key: str) -> bool:
        sidecar = out_dir / f"{key}.json"
        npz = sidecar.with_suffix(".npz")
        if args.overwrite or not (sidecar.is_file() and npz.is_file()):
            return False
        try:
            payload = json.loads(sidecar.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return False
        return payload.get("run_fingerprint") == run_fingerprint and payload.get("npz_bytes") == npz.stat().st_size

    todo = [k for k in shard_keys if not done(k)]
    already = len(shard_keys) - len(todo)
    LOGGER.info("shard plan: %d clips to run, %d already done, %d skipped", len(todo), already, len(skipped))

    shard_manifest_path = out_dir / "manifests" / f"shard-{args.shard_index:03d}-of-{args.num_shards:03d}.json"
    shard_manifest = {
        "schema": SCHEMA + "/shard",
        "run": run,
        "run_fingerprint": run_fingerprint,
        "shard": {"index": args.shard_index, "count": args.num_shards},
        "args": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "env": {k: os.environ.get(k) for k in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "INTERNNAV_MODEL_PATH",
                                               "MACA_PATH", "EXP18_ROOT", "EXP18_RENDER_ROOT")},
        "started_utc": utc_now(),
        "clips_planned": len(shard_keys),
        "clips_already_done": already,
        "clips_done": [],
        "clips_skipped": skipped,
        "clips_failed": failed,
    }
    write_json_atomic(shard_manifest_path, shard_manifest)
    if not todo:
        shard_manifest["finished_utc"] = utc_now()
        write_json_atomic(shard_manifest_path, shard_manifest)
        if args.num_shards == 1:
            merge_manifests(args)
        return 0

    # ---- dataset + row plan ------------------------------------------------
    t_import = time.time()
    import torch

    from src.data.amb3r_pose_cache import AMB3RPoseCache, AMB3RPoseCacheError

    cfg_probe = diagnostic_config(args)  # pulls in scripts.training (transformers): dominates cold start
    t_import = time.time() - t_import
    LOGGER.info("imports took %.1fs", t_import)
    t_ds = time.time()
    dataset = make_dataset([data_root / k for k in todo], data_root, cfg_probe)
    K = int(dataset.num_history_sample)
    pose_cache = AMB3RPoseCache(cache_root, dataset_root=data_root, num_history=K,
                                min_history=int(dataset.min_history), max_cached_clips=4)
    by_clip: dict = {}
    for ds_index, (clip_idx, t) in enumerate(dataset.sample_index):
        by_clip.setdefault(int(clip_idx), {})[int(t)] = ds_index
    items: list = []
    clip_endpoints: dict = {}
    for clip_idx, clip_dir in enumerate(dataset.clips):
        key = f"{clip_dir.parent.name}/{clip_dir.name}"
        frame_count = int(dataset._load_meta(clip_idx)["num_frames"])
        try:
            endpoints = pose_cache.current_frame_ids(clip_dir, expected_frame_count=frame_count)
        except AMB3RPoseCacheError as exc:
            # Recorded, never silently replaced by GT poses.
            if args.missing_cache == "error":
                raise
            failed.append({"clip": key, "reason": f"amb3r_cache_invalid: {exc}"})
            continue
        frames = by_clip.get(clip_idx, {})
        missing = [int(t) for t in endpoints if int(t) not in frames]
        if missing:
            raise RuntimeError(f"cache endpoint rows not indexable by the dataset in {key}: {missing[:8]}")
        clip_endpoints[clip_idx] = endpoints
        items.extend((frames[int(t)], clip_idx, int(t)) for t in endpoints)
    t_ds = time.time() - t_ds
    LOGGER.info("dataset built in %.1fs: %d clips, %d query rows", t_ds, len(dataset.clips), len(items))
    shard_manifest["timing_s"] = {"imports": round(t_import, 1), "dataset_index": round(t_ds, 1)}
    shard_manifest["rows_planned"] = len(items)
    if args.dry_list:
        print(json.dumps({"tier": args.tier, "clips": len(clip_endpoints), "rows": len(items),
                          "failed": failed, "skipped": skipped}, indent=2))
        return 0

    # ---- model ---------------------------------------------------------------
    cfg, model, collator, audit = load_everything(args)
    if int(cfg["data"]["sliding_window"]["num_history_sample"]) != K:
        raise RuntimeError("K mismatch between dataset and model config")
    run["head_audit"] = audit
    run["effective_heatmap_config"] = cfg["model"]["heatmap"]
    write_json_atomic(shard_manifest_path, shard_manifest)

    from scripts.training.utils import make_autocast_context

    device = torch.device(args.device)
    loader = torch.utils.data.DataLoader(
        DumpItems(dataset, items, pose_cache),
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=make_collate(collator),
        persistent_workers=False,
        prefetch_factor=4 if args.num_workers > 0 else None,
    )
    hm_size = tuple(cfg["model"]["heatmap"]["heatmap_size"])
    arms = run["arms"]

    def flush(buffer: ClipBuffer, clip_idx: int) -> None:
        meta = dataset._load_meta(clip_idx)
        clip_c2w = np.stack(dataset._load_poses(clip_idx)).astype(np.float32)
        arrays = buffer.arrays(args.tier, meta, clip_c2w, clip_endpoints[clip_idx])
        if len(arrays["current_frame_ids"]) != len(clip_endpoints[clip_idx]) or not np.array_equal(
                arrays["current_frame_ids"], clip_endpoints[clip_idx]):
            raise RuntimeError(f"rows of {buffer.scene}/{buffer.clip} do not equal the cache endpoints")
        npz = out_dir / buffer.scene / f"{buffer.clip}.npz"
        write_npz_atomic(npz, arrays)
        summary = buffer.summary(arrays)
        sidecar = {
            "schema": SCHEMA + "/clip",
            "tier": args.tier,
            "clip_key": f"{buffer.scene}/{buffer.clip}",
            "clip_dir": str(buffer.clip_dir),
            "npz": str(npz), "npz_bytes": npz.stat().st_size, "npz_sha256": sha256_file(npz),
            "run_fingerprint": run_fingerprint, "rows": summary["rows"],
            "summary": summary, "elapsed_s": round(time.time() - buffer.t0, 2), "written_utc": utc_now(),
        }
        write_json_atomic(npz.with_suffix(".json"), sidecar)
        shard_manifest["clips_done"].append(sidecar["clip_key"])
        LOGGER.info("wrote %s rows=%d %.1fs vo=%s gt=%s", sidecar["clip_key"], summary["rows"],
                    sidecar["elapsed_s"], summary["vo"], summary["gt"])

    current: ClipBuffer | None = None
    current_idx: int | None = None
    t_run = time.time()
    t_wait = t_fwd = t_write = 0.0
    t_mark = time.time()
    rows_done = 0
    self_checked = args.no_self_check
    for batch, extras in loader:
        t_wait += time.time() - t_mark
        t_mark = time.time()
        gt_rel = batch["history_rel_poses"]  # [B,k,4] (collator-padded)
        vo_rel = gt_rel.clone()
        for b, e in enumerate(extras):
            vo_rel[b].zero_()
            vo_rel[b, : e["vo_rel"].shape[0]] = torch.from_numpy(e["vo_rel"])
        with torch.inference_mode(), make_autocast_context(device, args.amp):
            outs = forward_arms(model, batch, {"vo": vo_rel, "gt": gt_rel})
            if not self_checked:
                ref = model(
                    video_frames=None,
                    single_view_inputs={"pixel_values": batch["pixel_values"].to(device),
                                        "image_grid_thw": batch["image_grid_thw"].to(device)},
                    single_view_num_histories=batch["num_histories"],
                    history_rel_poses=gt_rel.to(device),
                    return_heatmaps=True, return_heatmap_logits=True,
                    return_actions=False, return_lm_loss=False,
                )
                # MACA bf16 kernels are not run-to-run deterministic, so the
                # tolerance is a repeat of our own forward (noise floor).
                repeat = forward_arms(model, batch, {"gt": gt_rel})["gt"]

                def _maxdiff(a, b):
                    return float((a.float() - b.float()).abs().max())

                g_ref = _maxdiff(ref["heatmaps_gated"], outs["gt"]["heatmaps_gated"])
                v_ref = _maxdiff(ref["visibility"], outs["gt"]["visibility"])
                g_rep = _maxdiff(repeat["heatmaps_gated"], outs["gt"]["heatmaps_gated"])
                v_rep = _maxdiff(repeat["visibility"], outs["gt"]["visibility"])
                g_ok = g_ref <= max(1e-3, 4.0 * g_rep)
                v_ok = v_ref <= max(0.25, 4.0 * v_rep)
                run["self_check"] = {
                    "sample": extras[0]["sample_identity"],
                    "vs_model_forward": {"max_abs_diff_heatmaps_gated": g_ref, "max_abs_diff_visibility": v_ref},
                    "vs_repeat_split_forward": {"max_abs_diff_heatmaps_gated": g_rep,
                                                "max_abs_diff_visibility": v_rep},
                    "tolerance": "gated <= max(1e-3, 4*repeat); visibility logits <= max(0.25, 4*repeat)",
                    "passed": g_ok and v_ok,
                }
                LOGGER.info("self-check: %s", json.dumps(run["self_check"]))
                if not (g_ok and v_ok):
                    raise RuntimeError(f"split forward diverges from model.forward: {run['self_check']}")
                self_checked = True
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        t_fwd += time.time() - t_mark
        t_mark = time.time()
        mask = batch["history_mask"].cpu().numpy()
        for b, e in enumerate(extras):
            clip_idx = e["clip_idx"]
            if clip_idx != current_idx:
                if current is not None:
                    flush(current, current_idx)
                current = ClipBuffer(dataset.clips[clip_idx], K, hm_size, arms)
                current_idx = clip_idx
            current.add(e, outs, b, mask[b])
            rows_done += 1
        t_write += time.time() - t_mark
        t_mark = time.time()
        if rows_done % 200 == 0:
            LOGGER.info("rows %d/%d (%.2f rows/s)", rows_done, len(items),
                        rows_done / max(time.time() - t_run, 1e-6))
    if current is not None:
        flush(current, current_idx)

    elapsed = time.time() - t_run
    shard_manifest["timing_s"].update({"inference": round(elapsed, 1), "rows": rows_done,
                                       "rows_per_s": round(rows_done / max(elapsed, 1e-6), 3),
                                       "loader_wait": round(t_wait, 1), "forward": round(t_fwd, 1),
                                       "buffer_and_write": round(t_write, 1)})
    shard_manifest["finished_utc"] = utc_now()
    shard_manifest["run"] = run
    write_json_atomic(shard_manifest_path, shard_manifest)
    if args.num_shards == 1:
        merge_manifests(args)
    return 0


if __name__ == "__main__":
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    raise SystemExit(main())

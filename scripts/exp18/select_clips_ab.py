#!/usr/bin/env python3
"""EXP-18: write the pre-registered clip lists of tiers A and B (R2R v2).

Rules (docs/experiments/README.md, EXP-18, "层级"):

* Tier A, training scenes: the scene dirs under ``common.R2R_V2_ROOT`` minus
  ``common.R2R_V2_VAL_SCENES`` (asserted: 22).  Per scene, clip dirs sorted by
  ``sha1_key(f"{scene}/{clip}")`` ascending; walk that order and take the first
  ``TIERS['A']['clips_per_scene']`` (10) that have a valid AMB3R endpoint
  cache in ``common.R2R_V2_AMB3R_CACHE``.  Every clip passed over is reported
  (in the list header and on stdout).
* Tier B, held-out scenes: every clip of the 4 val scenes (asserted: 858);
  every one must have a valid cache.

"Valid cache" = the strict reader the dataset and the dump use
(``AMB3RPoseCache.current_frame_ids`` with the clip's meta ``num_frames``).
The val/train scene sets are checked against the dataset's own MD5 auto-split
(``VLNSlidingWindowDataset._enumerate_clips`` with split='val' / 'train'),
and the clip enumeration is taken from that same function, so both lists
index exactly what the dataset would.

Writes ``EXP_ROOT/clip_lists/{A,B}.txt`` via ``common.write_clip_list`` (the
header documents the rule and counts).  Needs the qwen25 env (the dataset
module imports torch); no GPU.

  $W/envs/qwen25/bin/python scripts/exp18/select_clips_ab.py
  EXP18_ROOT=/tmp/exp18_dev/x/exp18_root $W/envs/qwen25/bin/python scripts/exp18/select_clips_ab.py
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

SOURCE_ROOT = Path(__file__).resolve().parents[2]
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from scripts.exp18 import common  # noqa: E402

EXPECTED_TRAIN_SCENES = 22
EXPECTED_B_CLIPS = 858


def dataset_split(root: Path, split: str) -> list:
    """Clips the dataset's own auto-split assigns to ``split`` (no __init__ side effects)."""
    from src.data.sliding_window_dataset import VLNSlidingWindowDataset

    ds = VLNSlidingWindowDataset.__new__(VLNSlidingWindowDataset)
    ds.root, ds.split, ds.max_clip_id, ds.max_clips = Path(root), split, 0, 0
    return ds._enumerate_clips()


def cache_status(clip_dir: Path) -> str:
    """'ok' or the reason the strict reader rejects this clip's cache."""
    from src.data.amb3r_pose_cache import AMB3RPoseCache

    meta = clip_dir / "meta.json"
    if not meta.is_file():
        return "no_meta_json (empty placeholder)"
    try:
        frames = int(json.loads(meta.read_text(encoding="utf-8"))["num_frames"])
        reader = AMB3RPoseCache(common.R2R_V2_AMB3R_CACHE, dataset_root=common.R2R_V2_ROOT,
                                num_history=common.NUM_HISTORY, min_history=common.MIN_HISTORY,
                                max_cached_clips=1)
        reader.current_frame_ids(clip_dir, expected_frame_count=frames)
    except Exception as exc:  # noqa: BLE001 - every rejection is reported, not raised
        return f"{type(exc).__name__}: {exc}"
    return "ok"


def main(argv: list | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--workers", type=int, default=16, help="threads for cache validation (AFS-bound)")
    args = p.parse_args(argv)

    root = common.R2R_V2_ROOT
    val_scenes = sorted(common.R2R_V2_VAL_SCENES)
    all_scenes = sorted(d.name for d in root.iterdir() if d.is_dir())
    train_scenes = sorted(set(all_scenes) - set(val_scenes))
    assert set(val_scenes) <= set(all_scenes), f"val scenes missing under {root}"
    assert len(train_scenes) == EXPECTED_TRAIN_SCENES, f"expected 22 train scenes, got {len(train_scenes)}"

    ds_val = dataset_split(root, "val")
    ds_train = dataset_split(root, "train")
    ds_val_scenes = sorted({c.parent.name for c in ds_val})
    ds_train_scenes = sorted({c.parent.name for c in ds_train})
    assert ds_val_scenes == val_scenes, f"dataset MD5 split val scenes {ds_val_scenes} != {val_scenes}"
    assert ds_train_scenes == train_scenes, "dataset MD5 split train scenes != all minus val"

    # Tier A: per scene, sha1 order, first N with a valid cache (walked lazily).
    per_scene = int(common.TIERS["A"]["clips_per_scene"])
    by_scene: dict = {}
    for clip in ds_train:
        by_scene.setdefault(clip.parent.name, []).append(clip)

    def select_scene(scene: str) -> tuple:
        taken, passed = [], []
        for clip in sorted(by_scene[scene], key=lambda c: common.sha1_key(f"{scene}/{c.name}")):
            if len(taken) == per_scene:
                break
            reason = cache_status(clip)
            if reason == "ok":
                taken.append(f"{scene}/{clip.name}")
            else:
                passed.append({"clip": f"{scene}/{clip.name}", "reason": reason})
        return taken, passed

    with ThreadPoolExecutor(max_workers=max(1, args.workers)) as pool:
        a_results = list(pool.map(select_scene, train_scenes))
        b_status = list(pool.map(cache_status, ds_val))
    a_keys = [k for taken, _ in a_results for k in taken]
    a_passed = [row for _, passed in a_results for row in passed]
    for scene, (taken, _) in zip(train_scenes, a_results):
        assert len(taken) == per_scene, f"scene {scene}: only {len(taken)} clips with a valid cache"

    # Tier B: every clip of the 4 val scenes; all must have a valid cache.
    b_keys = [f"{c.parent.name}/{c.name}" for c in ds_val]
    b_bad = [{"clip": k, "reason": r} for k, r in zip(b_keys, b_status) if r != "ok"]
    b_per_scene = {s: sum(k.startswith(s + "/") for k in b_keys) for s in val_scenes}
    assert not b_bad, f"tier B clips without a valid cache: {b_bad[:5]}"
    assert len(b_keys) == EXPECTED_B_CLIPS, f"tier B: expected {EXPECTED_B_CLIPS} clips, got {len(b_keys)}"

    stamp = _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds")
    common_header = (
        f"generated {stamp} by scripts/exp18/select_clips_ab.py\n"
        f"data root {root}\n"
        f"cache root {common.R2R_V2_AMB3R_CACHE} (validity = strict AMB3RPoseCache reader, meta num_frames)\n"
        f"dataset MD5 auto-split (VLNSlidingWindowDataset._enumerate_clips) val scenes = {' '.join(ds_val_scenes)}\n"
    )
    a_header = common_header + (
        f"EXP-18 tier A (training scenes): {len(train_scenes)} scenes = all minus the 4 val scenes; per scene\n"
        f"clips sorted by sha1('<scene>/<clip>') ascending, first {per_scene} with a valid cache: {len(a_keys)} clips\n"
        f"passed over (no valid cache): {len(a_passed)}\n"
        + "".join(f"  passed over {row['clip']}: {row['reason'][:160]}\n" for row in a_passed)
    )
    b_header = common_header + (
        f"EXP-18 tier B (held-out scenes): every clip of {' '.join(val_scenes)}: {len(b_keys)} clips\n"
        f"per scene: {json.dumps(b_per_scene)}\n"
    )
    path_a = common.write_clip_list("A", a_keys, a_header)
    path_b = common.write_clip_list("B", b_keys, b_header)
    print(json.dumps({
        "A": {"path": str(path_a), "clips": len(a_keys), "scenes": len(train_scenes), "passed_over": a_passed},
        "B": {"path": str(path_b), "clips": len(b_keys), "per_scene": b_per_scene},
        "dataset_val_scenes": ds_val_scenes,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

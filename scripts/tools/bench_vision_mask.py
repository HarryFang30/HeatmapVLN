#!/usr/bin/env python
"""What Qwen2.5-VL's vision window mask costs per plan call, on this accelerator.

``src/models/qwen2_5_vl_vision_mask.py`` replaces the one line transformers 4.51.0
uses to build the vision window mask (``Qwen2_5_VLVisionSdpaAttention.forward``), and
it is measured on the 910B: 1026.5 ms -> 188.9 ms over 32 layers.  Those numbers do
not transfer to CUDA.  The cost the patch removes is host-side -- a one-element
device-to-host copy per slice bound, each a full synchronisation, plus one kernel
launch per window -- and a synchronisation costs about 120 us on the 910B against
10-20 us here.  So the saving on a 4090 has to be measured on a 4090, which is what
this does, without the 39 GB of accelerator memory a pair of real servers needs.

It measures the construction alone, which is the whole patch: the rest of
``forward`` -- qkv, rotary, SDPA, proj -- is untouched, so the difference between
two arms at the same shape is the difference the patch makes to the call.  Three arms:

    upstream_device   transformers' loop with the bounds still on the device
                      (what a server without the patch runs)
    upstream_host     the same loop with the bounds read once by .tolist()
                      (the first half of the patch: no scalar reads left)
    window_id         what the patch ships: four kernels, no per-window launch

The shapes are the deployed ones, and they are not guessed.  The client sends 384x384
front views and a 640x480 lookdown (``src/deploy/nav_agent.py``:84-86); the processor
rounds each side to a multiple of patch_size * spatial_merge_size = 28, giving 392x392
-> a 28x28 patch grid (784 patches) and 476x644 -> 34x46 (1564).  With K=8 that is
8 history + 1 current = 9 images (7056 patches) in turn 1 and one more in turn 2
(8620) -- exactly the shapes the server's own pass counter reported for a steady-state
trajectory+ppa call, ``['7056x9', '8620x10', '7056x9', '8620x10']``
(``src/models/qwen2_5_vl_vision_count.py``), hence the default four passes.

The window bounds come from upstream's own ``get_window_index``, called on a stub
carrying the four scalars it reads, so the window partition is the real one rather
than a uniform approximation: at this grid it is 16 windows per 28x28 image, not
seq_length / 64.  Per pass the mask is built once per vision layer, with the window
bounds in the windowed layers and the image bounds in the four full-attention blocks
(``fullatt_block_indexes``).

Usage::

    python scripts/tools/bench_vision_mask.py --device cuda:4 --repeats 9 \
        [--out bench.json] [--pass 9x28x28 --pass 9x28x28,1x34x46 ...]

The arms are measured interleaved, one repeat of each in turn, so that a neighbour's
load on a shared machine drifts across all three rather than into one of them; the
summary reports the median and the minimum, and the minimum is the one to read when
the machine is busy.
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[2]
if (REPO_ROOT / "src" / "models" / "qwen2_5_vl_vision_mask.py").is_file():
    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))

# The deployed vision tower, from the served model's config.json.  Passed in rather
# than hardcoded would be nicer, but a bench that reads the 15 GB model's config has
# to find the model; these four are checked against the config in the report instead.
DEPTH = 32
FULLATT_BLOCK_INDEXES = (7, 15, 23, 31)
PATCH_SIZE = 14
SPATIAL_MERGE_SIZE = 2
WINDOW_SIZE = 112

# One steady-state trajectory+ppa call, as the server's pass counter recorded it.
DEFAULT_PASSES = ("9x28x28", "9x28x28,1x34x46", "9x28x28", "9x28x28,1x34x46")


def parse_grids(spec: str) -> List[Tuple[int, int, int]]:
    """``"9x28x28,1x34x46"`` -> nine (1, 28, 28) grids then one (1, 34, 46)."""
    grids: List[Tuple[int, int, int]] = []
    for group in spec.split(","):
        count, _, grid = group.partition("x")
        h, _, w = grid.partition("x")
        grids.extend([(1, int(h), int(w))] * int(count))
    if not grids:
        raise ValueError(f"no grids in {spec!r}")
    return grids


def _stub_window_index(grid_thw: "Any") -> List[int]:
    """Upstream's ``get_window_index``, called on a stub instead of a 675M-parameter tower.

    The method reads exactly four scalars off ``self``, so binding it to a stub runs the
    real partition with no weights and no device memory -- the alternative, replicating
    the padding and permutation here, is a second copy of upstream logic to keep in step.
    """
    from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
        Qwen2_5_VisionTransformerPretrainedModel as Tower,
    )

    class _Stub:
        window_size = WINDOW_SIZE
        spatial_merge_size = SPATIAL_MERGE_SIZE
        patch_size = PATCH_SIZE
        spatial_merge_unit = SPATIAL_MERGE_SIZE * SPATIAL_MERGE_SIZE

    _, cu_window_seqlens = Tower.get_window_index(_Stub(), grid_thw)
    return cu_window_seqlens


def bounds_for(grids: Sequence[Tuple[int, int, int]], device: "Any") -> Dict[str, Any]:
    """The two bound vectors one pass builds masks from, as ``forward`` prepares them."""
    import torch

    grid_thw = torch.tensor(grids, dtype=torch.long)
    seq_length = int((grid_thw[:, 0] * grid_thw[:, 1] * grid_thw[:, 2]).sum())

    # Full attention: one span per image, forward's own expression.
    per_image = torch.repeat_interleave(grid_thw[:, 1] * grid_thw[:, 2], grid_thw[:, 0])
    full = torch.cat([torch.zeros(1, dtype=torch.long), per_image.cumsum(0)])

    # Windowed attention: upstream's partition, deduplicated the way forward does it.
    window = torch.unique_consecutive(torch.tensor(_stub_window_index(grid_thw), dtype=torch.long))

    return {
        "seq_length": seq_length,
        "images": len(grids),
        "full": full.to(device),
        "window": window.to(device),
        "n_full_spans": int(full.numel()) - 1,
        "n_windows": int(window.numel()) - 1,
    }


# ------------------------------------------------------------------------------ arms
def build_upstream(bounds: "Any", seq_length: int) -> "Any":
    """transformers 4.51.0's construction, bounds left on the device.

    ``bounds[i]`` in a slice calls ``__index__`` on a device tensor, so each iteration
    is two one-element device-to-host copies, and each of those drains the queue.
    """
    import torch

    mask = torch.zeros([1, seq_length, seq_length], device=bounds.device, dtype=torch.bool)
    for i in range(1, len(bounds)):
        mask[..., bounds[i - 1] : bounds[i], bounds[i - 1] : bounds[i]] = True
    return mask


def build_upstream_host(host_bounds: Sequence[int], seq_length: int, device: "Any") -> "Any":
    """The same loop with the bounds already on the host: the launches, none of the reads."""
    import torch

    mask = torch.zeros([1, seq_length, seq_length], device=device, dtype=torch.bool)
    for i in range(1, len(host_bounds)):
        mask[..., host_bounds[i - 1] : host_bounds[i], host_bounds[i - 1] : host_bounds[i]] = True
    return mask


def build_window_id(host_bounds: Sequence[int], seq_length: int, device: "Any") -> "Any":
    """What the patch ships, imported from the module the server installs."""
    from src.models.qwen2_5_vl_vision_mask import _window_mask

    return _window_mask(list(host_bounds), seq_length, device)


ARMS = ("upstream_device", "upstream_host", "window_id")


def time_once(arm: str, bounds: "Any", host: Sequence[int], seq_length: int, device: "Any") -> Tuple[float, "Any"]:
    """Seconds for one construction, with the device drained on both sides of it.

    Draining first means the clock does not charge this arm for work queued before it;
    draining after means an arm that only queues kernels is not credited for work it
    left pending.  The host-side synchronisations the patch removes are inside the
    window, which is the point.
    """
    import torch

    torch.cuda.synchronize(device)
    t0 = time.perf_counter()
    if arm == "upstream_device":
        mask = build_upstream(bounds, seq_length)
    elif arm == "upstream_host":
        mask = build_upstream_host(host, seq_length, device)
    else:
        mask = build_window_id(host, seq_length, device)
    torch.cuda.synchronize(device)
    return time.perf_counter() - t0, mask


def measure(bounds: "Any", seq_length: int, device: "Any", repeats: int) -> Dict[str, Any]:
    """Interleaved repeats of all three arms at one shape, plus the identity check."""
    import torch

    host = bounds.tolist()
    samples: Dict[str, List[float]] = {arm: [] for arm in ARMS}
    reference: Dict[str, Any] = {}

    for r in range(repeats + 1):  # repeat 0 is a warm-up and is discarded
        for arm in ARMS:
            seconds, mask = time_once(arm, bounds, host, seq_length, device)
            if r:
                samples[arm].append(seconds * 1000.0)
            if arm not in reference:
                reference[arm] = mask
            else:
                del mask
    identical = {
        arm: bool(torch.equal(reference[arm], reference["upstream_device"]))
        for arm in ARMS
    }
    for mask in reference.values():
        del mask
    torch.cuda.empty_cache()

    return {
        "ms": {
            arm: {
                "median": statistics.median(samples[arm]),
                "min": min(samples[arm]),
                "max": max(samples[arm]),
                "n": len(samples[arm]),
            }
            for arm in ARMS
        },
        "same_mask_as_upstream": identical,
    }


# ---------------------------------------------------------------------------- report
def per_pass_ms(shape: Dict[str, Any], stat: str) -> Dict[str, float]:
    """One pass: the mask is built once per layer, with whichever bounds that layer uses."""
    n_windowed = DEPTH - len(FULLATT_BLOCK_INDEXES)
    n_full = len(FULLATT_BLOCK_INDEXES)
    return {
        arm: n_windowed * shape["window"]["ms"][arm][stat] + n_full * shape["full"]["ms"][arm][stat]
        for arm in ARMS
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--repeats", type=int, default=9)
    ap.add_argument("--pass", dest="passes", action="append", default=None,
                    help="images of one vision-tower pass, e.g. 9x28x28,1x34x46 "
                         f"(default: the four passes of a ppa call, {' '.join(DEFAULT_PASSES)})")
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()

    import torch

    device = torch.device(args.device)
    if device.type != "cuda":
        raise SystemExit("this bench times host-side synchronisations against a CUDA device")
    torch.cuda.set_device(device)
    passes = list(args.passes or DEFAULT_PASSES)

    print(f"device {device} ({torch.cuda.get_device_name(device)}), torch {torch.__version__}")
    import transformers
    print(f"transformers {transformers.__version__}, "
          f"{DEPTH} layers, {len(FULLATT_BLOCK_INDEXES)} full-attention blocks")
    print(f"repeats {args.repeats} per arm per shape, interleaved\n")

    # One measurement per distinct shape; a repeated pass costs the same twice.
    shapes: Dict[str, Dict[str, Any]] = {}
    for spec in dict.fromkeys(passes):
        grids = parse_grids(spec)
        entry: Dict[str, Any] = {"spec": spec}
        for which in ("window", "full"):
            b = bounds_for(grids, device)
            entry["seq_length"] = b["seq_length"]
            entry["images"] = b["images"]
            entry[f"n_{which}"] = b["n_windows"] if which == "window" else b["n_full_spans"]
            entry[which] = measure(b[which], b["seq_length"], device, args.repeats)
            entry[which]["spans"] = entry[f"n_{which}"]
        shapes[spec] = entry
        print(f"{spec}: seq_length {entry['seq_length']}, {entry['images']} images, "
              f"{entry['n_window']} windows, {entry['n_full']} image spans")
        for which in ("window", "full"):
            for arm in ARMS:
                m = entry[which]["ms"][arm]
                same = entry[which]["same_mask_as_upstream"][arm]
                print(f"    {which:7s} {arm:16s} median {m['median']:8.2f} ms  "
                      f"min {m['min']:8.2f}  max {m['max']:8.2f}  same_mask={same}")
        print()

    print("per vision-tower pass (28 windowed layers + 4 full-attention layers), ms")
    for stat in ("median", "min"):
        print(f"  [{stat}]")
        for spec, entry in shapes.items():
            pp = per_pass_ms(entry, stat)
            print(f"    {spec:18s} " + "  ".join(f"{arm} {pp[arm]:8.1f}" for arm in ARMS))

    print("\nper plan call (" + " + ".join(passes) + "), ms")
    totals: Dict[str, Dict[str, float]] = {}
    for stat in ("median", "min"):
        total = {arm: 0.0 for arm in ARMS}
        for spec in passes:
            pp = per_pass_ms(shapes[spec], stat)
            for arm in ARMS:
                total[arm] += pp[arm]
        totals[stat] = total
        saved = total["upstream_device"] - total["window_id"]
        print(f"  [{stat}] " + "  ".join(f"{arm} {total[arm]:8.1f}" for arm in ARMS)
              + f"   saved {saved:8.1f}")

    all_same = all(
        entry[which]["same_mask_as_upstream"][arm]
        for entry in shapes.values() for which in ("window", "full") for arm in ARMS
    )
    print(f"\nevery arm built the same mask as upstream: {all_same}")

    if args.out:
        args.out.write_text(json.dumps({
            "device": str(device),
            "device_name": torch.cuda.get_device_name(device),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
            "depth": DEPTH,
            "fullatt_block_indexes": list(FULLATT_BLOCK_INDEXES),
            "repeats": args.repeats,
            "passes": passes,
            "shapes": {
                spec: {k: v for k, v in entry.items()}
                for spec, entry in shapes.items()
            },
            "per_call_ms": totals,
            "same_mask_everywhere": all_same,
        }, indent=2, default=str), encoding="utf-8")
        print(f"wrote {args.out}")


if __name__ == "__main__":
    main()

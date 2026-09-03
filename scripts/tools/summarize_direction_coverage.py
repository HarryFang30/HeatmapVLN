#!/usr/bin/env python3
"""Tabulate how history and future labels distribute over the four views.

EXP-11 replaces a "predict fewer directions" retraining ablation with a
statement about the labels themselves: what fraction of history waypoints is
visible in each canonical view, how many are visible in none of them, and how
concentrated the future segments are in the front view.  Both label families
are read through the production dataset, so the counts describe exactly the
supervision the heads receive.

Output is JSON plus a markdown table; the ledger cites the JSON path.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.config_schema import load_and_validate_config  # noqa: E402
from src.data.factory import build_dataset  # noqa: E402

VIEWS = ("front", "right", "back", "left")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="training config whose data section defines the labels")
    parser.add_argument("--split", default="val")
    parser.add_argument("--max-samples", type=int, default=400)
    parser.add_argument("--output-json", type=Path, default=None)
    args = parser.parse_args()

    cfg = load_and_validate_config(args.config)
    dataset = build_dataset(cfg, args.split)
    total = len(dataset)
    limit = min(args.max_samples, total) if args.max_samples > 0 else total

    hist_view_visible = [0] * 4
    hist_slots = 0
    hist_invisible_everywhere = 0
    hist_front_only = 0
    hist_without_front = 0
    fut_view_supported = [0] * 4
    fut_bins = 0
    fut_front_only = 0
    samples_used = 0
    samples_with_future = 0

    for index in range(limit):
        sample = dataset[index]
        visibility = sample.get("gt_visibility")
        if visibility is None:
            continue
        visibility = torch.as_tensor(visibility)
        if visibility.ndim != 2 or visibility.shape[-1] != 4:
            continue
        mask = sample.get("history_mask")
        if mask is not None:
            mask = torch.as_tensor(mask).reshape(-1).bool()
            visibility = visibility[mask[: visibility.shape[0]]]
        samples_used += 1
        visible = visibility > 0.5
        hist_slots += int(visible.shape[0])
        for view in range(4):
            hist_view_visible[view] += int(visible[:, view].sum())
        any_visible = visible.any(dim=-1)
        hist_invisible_everywhere += int((~any_visible).sum())
        hist_front_only += int((visible[:, 0] & ~visible[:, 1:].any(dim=-1)).sum())
        hist_without_front += int((any_visible & ~visible[:, 0]).sum())

        fut_vis = sample.get("future_trajectory_visibility")
        fut_mask = sample.get("future_trajectory_time_mask")
        if fut_vis is None or fut_mask is None:
            continue
        samples_with_future += 1
        fut_vis = torch.as_tensor(fut_vis) > 0.5
        fut_mask = torch.as_tensor(fut_mask).reshape(-1).bool()
        fut_vis = fut_vis[fut_mask[: fut_vis.shape[0]]]
        fut_bins += int(fut_vis.shape[0])
        for view in range(4):
            fut_view_supported[view] += int(fut_vis[:, view].sum())
        fut_front_only += int((fut_vis[:, 0] & ~fut_vis[:, 1:].any(dim=-1)).sum())

    def share(count: int, denominator: int) -> float:
        return float(count) / denominator if denominator else 0.0

    report = {
        "config": str(Path(args.config).resolve()),
        "split": args.split,
        "dataset_samples_total": total,
        "samples_scanned": limit,
        "samples_with_history_labels": samples_used,
        "samples_with_future_labels": samples_with_future,
        "history": {
            "slots": hist_slots,
            "visible_share_by_view": {
                view: share(hist_view_visible[i], hist_slots) for i, view in enumerate(VIEWS)
            },
            "invisible_in_all_views_share": share(hist_invisible_everywhere, hist_slots),
            "front_only_share": share(hist_front_only, hist_slots),
            "visible_but_not_in_front_share": share(hist_without_front, hist_slots),
        },
        "future": {
            "time_view_bins": fut_bins,
            "supported_share_by_view": {
                view: share(fut_view_supported[i], fut_bins) for i, view in enumerate(VIEWS)
            },
            "front_only_share": share(fut_front_only, fut_bins),
        },
    }

    print(f"# direction coverage ({args.split}, {samples_used} samples with history labels)")
    print("| family | front | right | back | left | none of the four |")
    print("|---|---|---|---|---|---|")
    hist = report["history"]["visible_share_by_view"]
    print(
        f"| history slots (n={hist_slots}) | "
        + " | ".join(f"{hist[view] * 100:.1f}%" for view in VIEWS)
        + f" | {report['history']['invisible_in_all_views_share'] * 100:.1f}% |"
    )
    fut = report["future"]["supported_share_by_view"]
    print(
        f"| future time-view bins (n={fut_bins}) | "
        + " | ".join(f"{fut[view] * 100:.1f}%" for view in VIEWS)
        + " | — |"
    )
    print(
        f"history visible only in front: {report['history']['front_only_share'] * 100:.1f}%; "
        f"visible but never in front: {report['history']['visible_but_not_in_front_share'] * 100:.1f}%"
    )
    print(f"future bins supported only in front: {report['future']['front_only_share'] * 100:.1f}%")

    if args.output_json is not None:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
        print(f"wrote {args.output_json}")


if __name__ == "__main__":
    main()

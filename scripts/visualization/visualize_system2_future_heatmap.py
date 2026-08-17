#!/usr/bin/env python3
"""Render System-2 future goals in the existing HeatmapVLN strip style.

The main strip uses the same F|R|B|L grouping, inferno colour map, black
inactive maps, yellow separators and fixed 64x64 heatmap convention as
``trajectory_heatmaps.py``.  It is a deterministic renderer preview, not a
trained model evaluation.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def _load_source_module(name: str, path: Path):
    """Load dependency-light source files without package eager imports."""

    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_pano = _load_source_module(
    "_future_vis_pano_view_pixel_goal",
    PROJECT_ROOT / "src/data/pano_view_pixel_goal.py",
)
_renderer = _load_source_module(
    "_future_vis_renderer",
    PROJECT_ROOT / "src/models/heatmap/future_heatmap_renderer.py",
)
PANO_HORIZONTAL_VIEWS = _pano.PANO_HORIZONTAL_VIEWS
label_clip_frames = _pano.label_clip_frames
load_intrinsics = _pano.load_intrinsics
FutureGoalEvidence = _renderer.FutureGoalEvidence
FutureHeatmapRenderer = _renderer.FutureHeatmapRenderer


VIEW_ORDER = tuple(PANO_HORIZONTAL_VIEWS)
SEP_COLOR = np.array([1.0, 0.85, 0.0], dtype=np.float32)


def _load_rgb(clip_dir: Path, frame_id: int, view: str) -> np.ndarray:
    key = f"rgb_{view}"
    for chunk_path in sorted((clip_dir / "chunks").glob("chunk_*.npz")):
        with np.load(chunk_path, allow_pickle=True) as archive:
            ids = np.asarray(archive["frame_ids"], dtype=np.int64)
            matches = np.flatnonzero(ids == int(frame_id))
            if matches.size == 0:
                continue
            payload = archive[key][int(matches[0])]
            if isinstance(payload, np.ndarray) and payload.ndim == 1:
                encoded = np.asarray(payload, dtype=np.uint8)
                bgr = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
                if bgr is None:
                    raise RuntimeError(f"Could not decode {key} at frame {frame_id}")
                return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            rgb = np.asarray(payload, dtype=np.uint8)
            if rgb.ndim != 3 or rgb.shape[-1] not in (3, 4):
                raise RuntimeError(f"Unexpected RGB shape {rgb.shape}")
            return rgb[..., :3]
    raise KeyError(f"frame {frame_id} missing from {clip_dir}")


def _resize_rgb(rgb: np.ndarray, tile: int) -> np.ndarray:
    return cv2.resize(rgb, (tile, tile), interpolation=cv2.INTER_AREA).astype(
        np.float32
    ) / 255.0


def _heatmap_rgb(heatmap: np.ndarray, tile: int) -> np.ndarray:
    if float(heatmap.max(initial=0.0)) <= 0.0:
        return np.zeros((tile, tile, 3), dtype=np.float32)
    resized = cv2.resize(
        np.asarray(heatmap, dtype=np.float32),
        (tile, tile),
        interpolation=cv2.INTER_LINEAR,
    )
    # Fixed range is mandatory: brightness must remain confidence.
    return plt.cm.inferno(np.clip(resized, 0.0, 1.0))[..., :3].astype(np.float32)


def _build_group_strip(
    records: list[dict[str, Any]],
    key: str,
    *,
    tile: int,
    gap: int,
) -> np.ndarray:
    group_width = 4 * tile
    width = len(records) * group_width + max(0, len(records) - 1) * gap
    strip = np.zeros((tile, width, 3), dtype=np.float32)
    for index, record in enumerate(records):
        x0 = index * (group_width + gap)
        for view_index in range(4):
            if key == "rgb":
                panel = _resize_rgb(record["rgb"][view_index], tile)
            elif key == "heatmap":
                panel = _heatmap_rgb(record["heatmaps"][view_index], tile)
            elif key == "overlay":
                base = _resize_rgb(record["rgb"][view_index], tile)
                hm = _heatmap_rgb(record["heatmaps"][view_index], tile)
                mask = cv2.resize(
                    record["heatmaps"][view_index],
                    (tile, tile),
                    interpolation=cv2.INTER_LINEAR,
                )[..., None]
                panel = np.clip(base * (1.0 - 0.55 * mask) + hm * (0.55 * mask), 0, 1)
            else:
                raise KeyError(key)
            strip[:, x0 + view_index * tile : x0 + (view_index + 1) * tile] = panel
        if index < len(records) - 1:
            strip[:, x0 + group_width : x0 + group_width + gap] = SEP_COLOR
    return strip


def _select_frame_ids(labels: dict[str, dict[str, Any]], count: int) -> list[int]:
    eligible = sorted(
        int(frame_id)
        for frame_id, entry in labels.items()
        if entry.get("eligible_sft") and entry.get("sample_kind") == "pixel"
    )
    if not eligible:
        raise RuntimeError("clip has no eligible future pixel goals")
    indices = np.unique(
        np.linspace(0, len(eligible) - 1, min(count, len(eligible)), dtype=np.int64)
    )
    return [eligible[int(index)] for index in indices]


def render_clip_strip(
    *,
    clip_dir: Path,
    output: Path,
    frame_count: int,
    confidence: float,
    tile: int,
    gap: int,
) -> dict[str, Any]:
    labels = label_clip_frames(clip_dir, img_size=(256, 256), min_history=5)
    frame_ids = _select_frame_ids(labels, frame_count)
    intrinsics = load_intrinsics(clip_dir)
    fx_256 = float(intrinsics["fx"]) * 256.0 / float(intrinsics["width"])
    renderer = FutureHeatmapRenderer(heatmap_size=(64, 64))
    records: list[dict[str, Any]] = []

    for frame_id in frame_ids:
        label = labels[str(frame_id)]
        pixel = label["pano_pixel_goal"]
        view = str(label["pano_view_id"])
        distance = float(label["waypoint_dist_m"])
        rendered = renderer.render(
            FutureGoalEvidence(
                pixel_uv=(float(pixel[0]), float(pixel[1])),
                source_image_size=(256, 256),
                coordinate_frame="panoramic",
                view_id=view,
                distance_m=distance,
                confidence=float(confidence),
                camera_fx_px=fx_256,
                pixel_goal_source="expert_future_waypoint_projection_preview",
                distance_source="camera_z_depth_m",
                confidence_source="explicit_visualization_constant",
                system2_call_id=f"{clip_dir.name}/frame-{frame_id}",
            )
        )
        records.append(
            {
                "frame_id": frame_id,
                "view": view,
                "distance_m": distance,
                "confidence": confidence,
                "sigma_px": rendered.sigma_px,
                "rgb": [_load_rgb(clip_dir, frame_id, name) for name in VIEW_ORDER],
                "heatmaps": rendered.heatmaps,
            }
        )

    strips = [
        (_build_group_strip(records, "rgb", tile=tile, gap=gap), "RGB (F|R|B|L)"),
        (
            _build_group_strip(records, "heatmap", tile=tile, gap=gap),
            "Future Heatmap",
        ),
        (_build_group_strip(records, "overlay", tile=tile, gap=gap), "Overlay"),
    ]
    dpi = 120
    width_px = strips[0][0].shape[1]
    fig = plt.figure(figsize=(max(width_px / dpi, 12), 3.25))
    grid = fig.add_gridspec(
        3,
        1,
        hspace=0.05,
        left=0.025,
        right=0.995,
        top=0.88,
        bottom=0.12,
    )
    title = (
        f"System2 Future Heatmap — {clip_dir.parent.name}/{clip_dir.name}  |  "
        f"fixed confidence={confidence:.2f} (visual semantics preview)  |  "
        "brightness=confidence, size=distance (near larger)"
    )
    fig.suptitle(title, fontsize=8, y=0.97)
    group_width = 4 * tile
    tick_x = [index * (group_width + gap) + group_width / 2 for index in range(len(records))]
    tick_labels = [
        f"t{record['frame_id']} {record['view'][0].upper()}\n"
        f"d={record['distance_m']:.1f}m σ={record['sigma_px']:.1f}"
        for record in records
    ]
    for row, (strip, label) in enumerate(strips):
        axis = fig.add_subplot(grid[row])
        axis.imshow(strip, interpolation="nearest", aspect="equal")
        axis.set_ylabel(label, fontsize=6, labelpad=8)
        axis.set_yticks([])
        axis.set_xticks(tick_x)
        axis.set_xticklabels(tick_labels if row == len(strips) - 1 else [], fontsize=4)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)

    report = {
        "schema": "heatmapvln-system2-future-strip-v1",
        "clip_dir": str(clip_dir),
        "output": str(output),
        "view_order": list(VIEW_ORDER),
        "heatmap_size": [64, 64],
        "colour_map": "inferno",
        "colour_range": [0.0, 1.0],
        "confidence": float(confidence),
        "confidence_source": "explicit_visualization_constant",
        "not_a_model_confidence_claim": True,
        "records": [
            {key: value for key, value in record.items() if key not in {"rgb", "heatmaps"}}
            for record in records
        ],
    }
    with open(output.with_suffix(".json"), "w") as handle:
        json.dump(report, handle, indent=2, ensure_ascii=False)
    return report


def render_semantics_panel(output: Path) -> None:
    renderer = FutureHeatmapRenderer(heatmap_size=(64, 64))
    confidences = (0.25, 0.5, 0.75, 1.0)
    distances = (1.5, 3.0, 5.0, 8.0)
    fig, axes = plt.subplots(2, 4, figsize=(10, 5.2), constrained_layout=True)
    for col, confidence in enumerate(confidences):
        rendered = renderer.render(
            FutureGoalEvidence(
                pixel_uv=(128, 128),
                source_image_size=(256, 256),
                coordinate_frame="panoramic",
                view_id="front",
                distance_m=3.0,
                confidence=confidence,
                camera_fx_px=155.2,
                pixel_goal_source="semantic_demo",
                distance_source="explicit_3m",
                confidence_source="explicit_sweep",
            )
        )
        axes[0, col].imshow(rendered.heatmaps[0], cmap="inferno", vmin=0, vmax=1)
        axes[0, col].set_title(f"c={confidence:.2f}, d=3m\nσ={rendered.sigma_px:.1f}px", fontsize=9)
    for col, distance in enumerate(distances):
        rendered = renderer.render(
            FutureGoalEvidence(
                pixel_uv=(128, 128),
                source_image_size=(256, 256),
                coordinate_frame="panoramic",
                view_id="front",
                distance_m=distance,
                confidence=0.85,
                camera_fx_px=155.2,
                pixel_goal_source="semantic_demo",
                distance_source="explicit_sweep",
                confidence_source="explicit_0.85",
            )
        )
        axes[1, col].imshow(rendered.heatmaps[0], cmap="inferno", vmin=0, vmax=1)
        axes[1, col].set_title(f"c=0.85, d={distance:g}m\nσ={rendered.sigma_px:.1f}px", fontsize=9)
    axes[0, 0].set_ylabel("Brightness sweep\n(distance fixed)", fontsize=10)
    axes[1, 0].set_ylabel("Size sweep\n(confidence fixed)", fontsize=10)
    for axis in axes.flat:
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        "Future Heatmap semantics (fixed colour scale 0–1): brightness = confidence; size = distance",
        fontsize=11,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160, bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--clip-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--semantics-output", type=Path, required=True)
    parser.add_argument("--frame-count", type=int, default=24)
    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument("--tile", type=int, default=64)
    parser.add_argument("--gap", type=int, default=4)
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be in [0,1]")
    render_clip_strip(
        clip_dir=args.clip_dir.resolve(strict=True),
        output=args.output,
        frame_count=max(1, int(args.frame_count)),
        confidence=float(args.confidence),
        tile=max(16, int(args.tile)),
        gap=max(1, int(args.gap)),
    )
    render_semantics_panel(args.semantics_output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

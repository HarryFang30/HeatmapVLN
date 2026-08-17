#!/usr/bin/env python3
"""Visualize native InternNav future goals using RGB-only AMB3R depth.

The recorded DAgger samples provide the native System-2 pixel goal and the
exact look-down RGB image seen by System-2.  Depth is predicted from those RGB
images by DA3NESTED-GIANT-LARGE.  Simulator/GT depth is never opened.

The layout intentionally follows the existing trajectory heatmap gallery:
yellow-separated temporal groups, F|R|B|L RGB context, an inferno heatmap row,
and an RGB overlay row.  Native InternNav coordinates live in the look-down
image, so the future heatmap has one honest ``front_down`` channel rather than
pretending that it is one of the four horizontal views.
"""

from __future__ import annotations

import argparse
import importlib.util
import io
import json
import math
import sys
import tarfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
VIEW_ORDER = ("front", "right", "back", "left")
SEP_COLOR = np.array([1.0, 0.85, 0.0], dtype=np.float32)
REPORT_SCHEMA = "heatmapvln-system2-future-amb3r-rgb-only-v1"
DEPTH_SOURCE = "amb3r_da3nested_giant_large_metric_depth_from_rgb"


def _load_source_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Could not load {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


_renderer = _load_source_module(
    "_future_amb3r_renderer",
    PROJECT_ROOT / "src/models/heatmap/future_heatmap_renderer.py",
)
FutureGoalEvidence = _renderer.FutureGoalEvidence
FutureHeatmapRenderer = _renderer.FutureHeatmapRenderer
metric_depth_at_pixel = _renderer.metric_depth_at_pixel
pinhole_intrinsics_from_hfov = _renderer.pinhole_intrinsics_from_hfov
project_lookdown_pixel_to_front = _renderer.project_lookdown_pixel_to_front


@dataclass
class NativeGoalRecord:
    episode_key: str
    frame_id: int
    system2_call_index: int
    llm_output: str
    native_pixel_goal_yx: tuple[int, int]
    pixel_uv: tuple[int, int]
    lookdown_rgb: np.ndarray
    panorama_rgb: tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    predicted_depth_m: np.ndarray | None = None
    predicted_depth_conf: np.ndarray | None = None
    prediction_is_metric: bool = False
    prediction_scale_factor: float | None = None
    depth_at_goal_m: float | None = None
    depth_conf_at_goal: float | None = None
    front_projection_valid: bool = False
    front_projection_reason: str | None = None
    front_pixel_uv: tuple[float, float] | None = None
    front_z_depth_m: float | None = None
    raw_elevation_delta_m: float | None = None
    used_elevation_delta_m: float | None = None
    height_mode: str | None = None
    heatmap: np.ndarray | None = None
    sigma_px: float | None = None


def _read_rgb(member: tarfile.TarInfo, archive: tarfile.TarFile) -> np.ndarray:
    handle = archive.extractfile(member)
    if handle is None:
        raise RuntimeError(f"Could not read {member.name}")
    with Image.open(io.BytesIO(handle.read())) as image:
        return np.asarray(image.convert("RGB"), dtype=np.uint8)


def _read_jsonl(archive: tarfile.TarFile, name: str) -> list[dict[str, Any]]:
    handle = archive.extractfile(name)
    if handle is None:
        raise FileNotFoundError(f"{name} is missing from {archive.name}")
    return [
        json.loads(line)
        for line in handle.read().decode("utf-8").splitlines()
        if line.strip()
    ]


def load_native_goal_records(
    episode_tar: Path,
    *,
    max_records: int,
) -> list[NativeGoalRecord]:
    """Load only RGB plus recorded native System-2 output from one episode."""

    with tarfile.open(episode_tar, mode="r") as archive:
        episode = json.loads(archive.extractfile("episode.json").read().decode("utf-8"))
        episode_key = str(episode["episode_key"])
        frames = {
            int(row["frame_id"]): row
            for row in _read_jsonl(archive, "frames.jsonl")
        }
        candidates: list[dict[str, Any]] = []
        for sample in _read_jsonl(archive, "samples.jsonl"):
            native = sample.get("native") or {}
            pixel = native.get("pixel_goal")
            frame_id = int(sample.get("current_frame_id", -1))
            frame = frames.get(frame_id)
            if (
                str(sample.get("native_kind", native.get("kind", "")))
                != "trajectory"
                or not isinstance(pixel, list)
                or len(pixel) != 2
                or frame is None
                or not frame.get("lookdown")
            ):
                continue
            candidates.append(sample)
        candidates.sort(
            key=lambda sample: (
                int(sample.get("system2_call_index", 0)),
                int(sample["current_frame_id"]),
            )
        )
        if max_records > 0:
            candidates = candidates[:max_records]
        records: list[NativeGoalRecord] = []
        for sample in candidates:
            frame_id = int(sample["current_frame_id"])
            frame = frames[frame_id]
            native = sample["native"]
            native_pixel_yx = tuple(int(value) for value in native["pixel_goal"])
            # Native InternNav stores [row, col].  Its pixel_to_gps helper does
            # ``v, u = pixel`` before indexing depth[v, u].  Convert once at
            # this boundary so the renderer consistently sees (u, v).
            lookdown_member = archive.getmember(str(frame["lookdown"]))
            lookdown = _read_rgb(lookdown_member, archive)
            width, height = lookdown.shape[1], lookdown.shape[0]
            pixel = native_pixel_yx_to_uv(native_pixel_yx, (width, height))
            panorama = tuple(
                _read_rgb(archive.getmember(str(frame["views"][view])), archive)
                for view in VIEW_ORDER
            )
            records.append(
                NativeGoalRecord(
                    episode_key=episode_key,
                    frame_id=frame_id,
                    system2_call_index=int(sample.get("system2_call_index", 0)),
                    llm_output=str(native.get("llm_output", "")),
                    native_pixel_goal_yx=native_pixel_yx,
                    pixel_uv=pixel,
                    lookdown_rgb=lookdown,
                    panorama_rgb=panorama,
                )
            )
    if not records:
        raise RuntimeError(f"No native pixel-goal records found in {episode_tar}")
    return records


def _resolve_metric_flag(value: Any) -> bool:
    if hasattr(value, "item"):
        value = value.item()
    return int(value) == 1


def _resolve_optional_float(value: Any) -> float | None:
    if value is None:
        return None
    if hasattr(value, "item"):
        value = value.item()
    resolved = float(value)
    return resolved if math.isfinite(resolved) else None


def native_pixel_yx_to_uv(
    pixel_yx: tuple[int, int] | list[int],
    image_size: tuple[int, int],
) -> tuple[int, int]:
    """Convert InternNav's stored [row, col] into renderer (u, v)."""

    if len(pixel_yx) != 2:
        raise ValueError("native pixel goal must have exactly two values")
    row, col = (int(pixel_yx[0]), int(pixel_yx[1]))
    width, height = (int(image_size[0]), int(image_size[1]))
    if width <= 0 or height <= 0:
        raise ValueError("image_size must be positive")
    if not (0 <= row < height and 0 <= col < width):
        raise ValueError(
            f"Native pixel [row={row}, col={col}] is outside {(width, height)}"
        )
    return col, row


def load_amb3r_depth_model(
    *, amb3r_repo: Path, checkpoint: Path, device: str
):
    """Load the shared DA3 model once for all requested episodes."""
    sys.path.insert(0, str(amb3r_repo))
    from amb3r.model_zoo import load_model

    wrapper = load_model("da3", ckpt_path=str(checkpoint))
    wrapper.device = device
    wrapper.model.to(device).eval()
    return wrapper.model


def predict_amb3r_depth(
    records: list[NativeGoalRecord],
    *,
    model: Any,
    process_res: int,
) -> dict[str, Any]:
    """Run one RGB-only DA3 inference and attach metric depth to records."""

    prediction = model.inference(
        [record.lookdown_rgb for record in records],
        process_res=int(process_res),
        process_res_method="upper_bound_resize",
        ref_view_strategy="first",
        infer_gs=False,
    )
    if not _resolve_metric_flag(prediction.is_metric):
        raise RuntimeError(
            "DA3 prediction is not marked metric; refusing to label point size as distance"
        )
    depth = np.asarray(prediction.depth, dtype=np.float32)
    if depth.ndim != 3 or depth.shape[0] != len(records):
        raise RuntimeError(
            f"Unexpected DA3 depth shape {depth.shape} for {len(records)} images"
        )
    confidence = (
        None
        if prediction.conf is None
        else np.asarray(prediction.conf, dtype=np.float32)
    )
    if confidence is not None and confidence.shape != depth.shape:
        raise RuntimeError(
            f"DA3 confidence/depth shape mismatch: {confidence.shape} vs {depth.shape}"
        )
    scale_factor = _resolve_optional_float(prediction.scale_factor)
    for index, record in enumerate(records):
        source_height, source_width = record.lookdown_rgb.shape[:2]
        aligned_depth = cv2.resize(
            depth[index],
            (source_width, source_height),
            interpolation=cv2.INTER_LINEAR,
        ).astype(np.float32)
        record.predicted_depth_m = aligned_depth
        record.prediction_is_metric = True
        record.prediction_scale_factor = scale_factor
        record.depth_at_goal_m = metric_depth_at_pixel(
            aligned_depth,
            record.pixel_uv,
            (source_width, source_height),
            neighborhood_radius=2,
        )
        if confidence is not None:
            aligned_conf = cv2.resize(
                confidence[index],
                (source_width, source_height),
                interpolation=cv2.INTER_LINEAR,
            ).astype(np.float32)
            record.predicted_depth_conf = aligned_conf
            u, v = record.pixel_uv
            patch = aligned_conf[
                max(0, v - 2) : min(source_height, v + 3),
                max(0, u - 2) : min(source_width, u + 3),
            ]
            valid = patch[np.isfinite(patch)]
            record.depth_conf_at_goal = (
                float(np.median(valid)) if valid.size else None
            )
    return {
        "prediction_is_metric": True,
        "prediction_scale_factor": scale_factor,
        "prediction_depth_shape": list(depth.shape),
        "process_res": int(process_res),
        "process_res_method": "upper_bound_resize",
    }


def render_future_heatmaps(
    records: list[NativeGoalRecord],
    *,
    confidence: float,
) -> None:
    renderer = FutureHeatmapRenderer(heatmap_size=(64, 64))
    for record in records:
        if record.depth_at_goal_m is None:
            raise RuntimeError("AMB3R depth must be predicted before rendering")
        height, width = record.lookdown_rgb.shape[:2]
        front_height, front_width = record.panorama_rgb[0].shape[:2]
        lookdown_k = pinhole_intrinsics_from_hfov(
            (width, height), hfov_degrees=90.0
        )
        front_k = pinhole_intrinsics_from_hfov(
            (front_width, front_height), hfov_degrees=90.0
        )
        projection = project_lookdown_pixel_to_front(
            pixel_uv_lookdown=(
                float(record.pixel_uv[0]),
                float(record.pixel_uv[1]),
            ),
            z_depth_lookdown_m=float(record.depth_at_goal_m),
            lookdown_intrinsics=lookdown_k,
            front_intrinsics=front_k,
            front_image_size=(front_width, front_height),
            lookdown_pitch_degrees=30.0,
            agent_height_m=1.25,
            flat_elevation_tolerance_m=0.20,
        )
        record.front_projection_valid = bool(projection.valid)
        record.front_projection_reason = projection.reason
        record.front_pixel_uv = projection.pixel_uv_front
        record.front_z_depth_m = projection.z_depth_front_m
        record.raw_elevation_delta_m = projection.raw_elevation_delta_m
        record.used_elevation_delta_m = projection.used_elevation_delta_m
        record.height_mode = projection.height_mode
        if not projection.valid:
            record.heatmap = np.zeros((4, 64, 64), dtype=np.float32)
            record.sigma_px = None
            continue
        if projection.pixel_uv_front is None or projection.z_depth_front_m is None:
            raise RuntimeError("valid front projection is missing geometry")
        rendered = renderer.render(
            FutureGoalEvidence(
                pixel_uv=projection.pixel_uv_front,
                source_image_size=(front_width, front_height),
                coordinate_frame="panoramic",
                view_id="front",
                distance_m=float(projection.z_depth_front_m),
                confidence=float(confidence),
                camera_fx_px=float(front_k[0, 0]),
                pixel_goal_source=(
                    "recorded_native_internnav_system2_output_geometrically_"
                    "projected_from_lookdown_to_front"
                ),
                distance_source=(
                    DEPTH_SOURCE + "_rotated_to_horizontal_front_z_depth"
                ),
                confidence_source="explicit_visualization_constant_not_model_score",
                system2_call_id=(
                    f"{record.episode_key}/call-{record.system2_call_index}"
                ),
            )
        )
        record.heatmap = rendered.heatmaps
        record.sigma_px = float(rendered.sigma_px)


def _resize_rgb(rgb: np.ndarray, width: int, height: int) -> np.ndarray:
    return cv2.resize(rgb, (width, height), interpolation=cv2.INTER_AREA).astype(
        np.float32
    ) / 255.0


def _heatmap_rgb(heatmap: np.ndarray, width: int, height: int) -> np.ndarray:
    resized = cv2.resize(
        np.asarray(heatmap, dtype=np.float32),
        (width, height),
        interpolation=cv2.INTER_LINEAR,
    )
    return plt.cm.inferno(np.clip(resized, 0.0, 1.0))[..., :3].astype(np.float32)


def _letterbox(rgb: np.ndarray, width: int, height: int) -> tuple[np.ndarray, tuple[int, int, int, int]]:
    src_h, src_w = rgb.shape[:2]
    scale = min(width / src_w, height / src_h)
    out_w = max(1, int(round(src_w * scale)))
    out_h = max(1, int(round(src_h * scale)))
    panel = np.zeros((height, width, 3), dtype=np.float32)
    resized = _resize_rgb(rgb, out_w, out_h)
    x0 = (width - out_w) // 2
    y0 = (height - out_h) // 2
    panel[y0 : y0 + out_h, x0 : x0 + out_w] = resized
    return panel, (x0, y0, out_w, out_h)


def _native_input_panel(record: NativeGoalRecord, width: int, height: int) -> np.ndarray:
    panel, (x0, y0, out_w, out_h) = _letterbox(record.lookdown_rgb, width, height)
    image = Image.fromarray(np.clip(panel * 255.0, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    src_h, src_w = record.lookdown_rgb.shape[:2]
    u = x0 + record.pixel_uv[0] * out_w / src_w
    v = y0 + record.pixel_uv[1] * out_h / src_h
    radius = max(3, height // 18)
    draw.ellipse(
        (u - radius, v - radius, u + radius, v + radius),
        outline=(0, 255, 255),
        width=max(1, radius // 2),
    )
    return np.asarray(image, dtype=np.float32) / 255.0


def _panoramic_heatmap_panel(
    record: NativeGoalRecord,
    width: int,
    height: int,
    *,
    overlay: bool,
) -> np.ndarray:
    if record.heatmap is None:
        raise RuntimeError("Future heatmap has not been rendered")
    if record.heatmap.shape != (4, 64, 64):
        raise RuntimeError(
            f"Expected four-view heatmap, got {record.heatmap.shape}"
        )
    tile_width = width // 4
    panels = []
    for view_index, rgb in enumerate(record.panorama_rgb):
        heat = _heatmap_rgb(
            record.heatmap[view_index], tile_width, height
        )
        if not overlay:
            panels.append(heat)
            continue
        base = _resize_rgb(rgb, tile_width, height)
        mask = cv2.resize(
            record.heatmap[view_index],
            (tile_width, height),
            interpolation=cv2.INTER_LINEAR,
        )[..., None]
        panels.append(
            np.clip(
                base * (1.0 - 0.62 * mask) + heat * (0.62 * mask),
                0.0,
                1.0,
            )
        )
    return np.concatenate(panels, axis=1)


def _depth_panel(record: NativeGoalRecord, width: int, height: int) -> np.ndarray:
    if record.predicted_depth_m is None:
        raise RuntimeError("Predicted depth is missing")
    depth = record.predicted_depth_m
    valid = depth[np.isfinite(depth) & (depth > 0.0)]
    if valid.size == 0:
        raise RuntimeError("Predicted depth has no valid values")
    low, high = np.quantile(valid, [0.02, 0.98])
    normalized = np.clip((depth - low) / max(float(high - low), 1e-6), 0.0, 1.0)
    colour = plt.cm.viridis(normalized)[..., :3].astype(np.float32)
    panel, _ = _letterbox(
        np.clip(colour * 255.0, 0, 255).astype(np.uint8), width, height
    )
    return panel


def _build_strip(
    records: list[NativeGoalRecord],
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
        if key == "panorama":
            group = np.concatenate(
                [_resize_rgb(rgb, tile, tile) for rgb in record.panorama_rgb],
                axis=1,
            )
        elif key == "system2_input":
            group = _native_input_panel(record, group_width, tile)
        elif key == "depth":
            group = _depth_panel(record, group_width, tile)
        elif key == "heatmap":
            group = _panoramic_heatmap_panel(
                record, group_width, tile, overlay=False
            )
        elif key == "overlay":
            group = _panoramic_heatmap_panel(
                record, group_width, tile, overlay=True
            )
        else:
            raise KeyError(key)
        strip[:, x0 : x0 + group_width] = group
        if index < len(records) - 1:
            strip[:, x0 + group_width : x0 + group_width + gap] = SEP_COLOR
    return strip


def render_episode_strip(
    records: list[NativeGoalRecord],
    *,
    output: Path,
    confidence: float,
    tile: int,
    gap: int,
    inference_audit: dict[str, Any],
    episode_tar: Path,
) -> dict[str, Any]:
    rows = [
        ("panorama", "RGB (F|R|B|L)"),
        ("system2_input", "System2 RGB↓ + goal"),
        ("depth", "AMB3R depth↓"),
        ("heatmap", "Future HM (F|R|B|L)"),
        ("overlay", "Overlay (F|R|B|L)"),
    ]
    strips = [
        (_build_strip(records, key, tile=tile, gap=gap), label)
        for key, label in rows
    ]
    dpi = 120
    width_px = strips[0][0].shape[1]
    figure = plt.figure(figsize=(max(width_px / dpi, 11), 5.1))
    grid = figure.add_gridspec(
        len(rows),
        1,
        hspace=0.035,
        left=0.035,
        right=0.995,
        top=0.88,
        bottom=0.13,
    )
    figure.suptitle(
        "Native System2 Future Heatmap — RGB only  |  lookdown pixel + "
        "AMB3R depth → 1.25m agent-height target (stairs preserved) → Front",
        fontsize=9,
        y=0.97,
    )
    group_width = 4 * tile
    tick_x = [
        index * (group_width + gap) + group_width / 2
        for index in range(len(records))
    ]
    tick_labels = [
        f"t{record.frame_id} call{record.system2_call_index} "
        f"uv↓={record.pixel_uv}\n"
        + (
            f"uvF=({record.front_pixel_uv[0]:.0f},"
            f"{record.front_pixel_uv[1]:.0f}) "
            f"d-hat={record.front_z_depth_m:.2f}m sigma={record.sigma_px:.1f}px "
            + (
                "flat@1.25m"
                if record.height_mode == "flat_agent_height_snapped"
                else f"delta-h={record.used_elevation_delta_m:+.2f}m"
            )
            if record.front_projection_valid
            and record.front_pixel_uv is not None
            and record.front_z_depth_m is not None
            and record.sigma_px is not None
            else "Front not visible"
        )
        for record in records
    ]
    for row_index, (strip, label) in enumerate(strips):
        axis = figure.add_subplot(grid[row_index])
        axis.imshow(strip, interpolation="nearest", aspect="equal")
        axis.set_ylabel(label, fontsize=6, labelpad=8)
        axis.set_yticks([])
        axis.set_xticks(tick_x)
        axis.set_xticklabels(
            tick_labels if row_index == len(rows) - 1 else [], fontsize=5
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.08)
    plt.close(figure)

    report = {
        "schema": REPORT_SCHEMA,
        "episode_tar": str(episode_tar),
        "episode_key": records[0].episode_key,
        "output": str(output),
        "input_modalities": ["rgb"],
        "system2_pixel_goal_source": "recorded_native_internnav_output",
        "depth_source": DEPTH_SOURCE,
        "gt_depth_read": False,
        "gt_pose_used_for_rendering": False,
        "future_label_used_for_rendering": False,
        "heatmap_coordinate_frame": "horizontal_panoramic",
        "heatmap_channel_order": list(VIEW_ORDER),
        "active_view_when_visible": "front",
        "lookdown_pitch_degrees": 30.0,
        "agent_height_m": 1.25,
        "flat_elevation_tolerance_m": 0.20,
        "height_policy": (
            "surface_point_is_lifted_by_1.25m; absolute elevation residuals_"
            "within_0.20m_are_snapped_to_current_agent_height; larger_up_or_"
            "down_changes_are_preserved"
        ),
        "projection": (
            "lookdown_uv_plus_amb3r_metric_z_depth_backprojected_to_3d_"
            "then_rotated_30deg_to_horizontal_front_and_reprojected"
        ),
        "projection_clamps_out_of_view_points": False,
        "heatmap_size": [64, 64],
        "heatmap_colour_map": "inferno",
        "heatmap_colour_range": [0.0, 1.0],
        "brightness_semantics": "explicit_constant_preview",
        "brightness_value": float(confidence),
        "brightness_limitation": (
            "recorded DAgger output has no token log-probabilities; brightness "
            "is not claimed as calibrated System2 confidence"
        ),
        "size_semantics": "near_larger_far_smaller_from_amb3r_metric_depth",
        "inference": inference_audit,
        "records": [
            {
                "frame_id": record.frame_id,
                "system2_call_index": record.system2_call_index,
                "llm_output": record.llm_output,
                "native_pixel_goal_yx": list(record.native_pixel_goal_yx),
                "pixel_uv": list(record.pixel_uv),
                "lookdown_image_size": [
                    int(record.lookdown_rgb.shape[1]),
                    int(record.lookdown_rgb.shape[0]),
                ],
                "amb3r_depth_at_goal_m": record.depth_at_goal_m,
                "amb3r_depth_conf_at_goal_raw": record.depth_conf_at_goal,
                "front_projection_valid": record.front_projection_valid,
                "front_projection_reason": record.front_projection_reason,
                "front_pixel_uv": (
                    list(record.front_pixel_uv)
                    if record.front_pixel_uv is not None
                    else None
                ),
                "front_z_depth_m": record.front_z_depth_m,
                "raw_elevation_delta_m": record.raw_elevation_delta_m,
                "used_elevation_delta_m": record.used_elevation_delta_m,
                "height_mode": record.height_mode,
                "sigma_px": record.sigma_px,
            }
            for record in records
        ],
    }
    output.with_suffix(".json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return report


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--episode-tar", type=Path, action="append", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--amb3r-repo", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--max-records", type=int, default=8)
    parser.add_argument("--confidence", type=float, default=0.85)
    parser.add_argument("--process-res", type=int, default=504)
    parser.add_argument("--tile", type=int, default=80)
    parser.add_argument("--gap", type=int, default=4)
    args = parser.parse_args()
    if not 0.0 <= args.confidence <= 1.0:
        raise ValueError("--confidence must be in [0,1]")
    if args.process_res <= 0 or args.tile < 32 or args.gap < 1:
        raise ValueError("invalid process-res/tile/gap")
    amb3r_repo = args.amb3r_repo.resolve(strict=True)
    checkpoint = args.checkpoint.resolve(strict=True)
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)

    # Load the giant model exactly once, then infer each episode independently
    # so camera/scene state never crosses episode boundaries.
    model = load_amb3r_depth_model(
        amb3r_repo=amb3r_repo,
        checkpoint=checkpoint,
        device=args.device,
    )
    all_reports = []
    for episode_tar_raw in args.episode_tar:
        episode_tar = episode_tar_raw.resolve(strict=True)
        records = load_native_goal_records(
            episode_tar, max_records=max(1, int(args.max_records))
        )
        inference = predict_amb3r_depth(
            records,
            model=model,
            process_res=int(args.process_res),
        )
        render_future_heatmaps(records, confidence=float(args.confidence))
        output = output_root / f"{records[0].episode_key}_future_heatmap_amb3r.png"
        all_reports.append(
            render_episode_strip(
                records,
                output=output,
                confidence=float(args.confidence),
                tile=int(args.tile),
                gap=int(args.gap),
                inference_audit=inference,
                episode_tar=episode_tar,
            )
        )
    (output_root / "manifest.json").write_text(
        json.dumps(
            {
                "schema": REPORT_SCHEMA,
                "gt_depth_read": False,
                "input_modalities": ["rgb"],
                "outputs": [report["output"] for report in all_reports],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"status": "ok", "outputs": len(all_reports)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Deterministic System-2 future-goal heatmap rendering.

This module deliberately contains no learnable parameters.  A System-2 pixel
goal already fixes the heatmap centre; metric depth fixes the Gaussian size;
and a separately supplied confidence fixes its peak.  Keeping those factors
separate gives the rendered map an auditable visual meaning:

* peak/brightness = confidence;
* Gaussian size = distance (near is larger, far is smaller);
* centre = the exact System-2 pixel goal.

Native InternNav pixel goals live in the ``front_down`` / look-down image, not
in the horizontal panorama.  The renderer therefore emits one channel for
native goals and four channels only for an explicitly panoramic goal.
"""

from __future__ import annotations

import io
import math
from dataclasses import dataclass
from typing import Any

import numpy as np


FUTURE_HEATMAP_SCHEMA_VERSION = "heatmapvln-future-goal-heatmap-v1"
FUTURE_HEATMAP_BLOB_NAME = "future_heatmap/heatmaps.npz"
FUTURE_HEATMAP_BLOB_MIME = "application/x-npz"
PANORAMIC_VIEW_ORDER = ("front", "right", "back", "left")
FRONT_DOWN_VIEW = "front_down"
LOOKDOWN_TO_FRONT_SCHEMA_VERSION = "heatmapvln-lookdown-to-front-projection-v1"


class FutureHeatmapContractError(ValueError):
    """Raised when evidence cannot support the promised heatmap semantics."""


@dataclass(frozen=True)
class LookdownToFrontProjection:
    """Result of deterministic look-down pixel to horizontal-front projection."""

    valid: bool
    reason: str | None
    pixel_uv_front: tuple[float, float] | None
    surface_point_front_xyz_m: tuple[float, float, float] | None
    point_front_xyz_m: tuple[float, float, float] | None
    z_depth_front_m: float | None
    source_pixel_uv: tuple[float, float]
    source_depth_m: float
    lookdown_pitch_degrees: float
    agent_height_m: float
    raw_elevation_delta_m: float | None
    used_elevation_delta_m: float | None
    flat_elevation_tolerance_m: float
    height_mode: str

    def metadata(self) -> dict[str, Any]:
        return {
            "schema": LOOKDOWN_TO_FRONT_SCHEMA_VERSION,
            "valid": bool(self.valid),
            "reason": self.reason,
            "pixel_uv_front": (
                list(self.pixel_uv_front)
                if self.pixel_uv_front is not None
                else None
            ),
            "surface_point_front_xyz_m": (
                list(self.surface_point_front_xyz_m)
                if self.surface_point_front_xyz_m is not None
                else None
            ),
            "point_front_xyz_m": (
                list(self.point_front_xyz_m)
                if self.point_front_xyz_m is not None
                else None
            ),
            "z_depth_front_m": self.z_depth_front_m,
            "source_pixel_uv": list(self.source_pixel_uv),
            "source_depth_m": float(self.source_depth_m),
            "lookdown_pitch_degrees": float(self.lookdown_pitch_degrees),
            "agent_height_m": float(self.agent_height_m),
            "raw_elevation_delta_m": self.raw_elevation_delta_m,
            "used_elevation_delta_m": self.used_elevation_delta_m,
            "flat_elevation_tolerance_m": float(
                self.flat_elevation_tolerance_m
            ),
            "height_mode": self.height_mode,
            "projection_semantics": (
                "lookdown_pixel_plus_metric_z_depth_backprojected_then_"
                "rotated_to_horizontal_front_camera_then_lifted_to_"
                "agent_height_with_flat_snap_and_height_change_preservation"
            ),
        }


@dataclass(frozen=True)
class FutureGoalEvidence:
    """All evidence needed to render one semantically valid future goal.

    ``pixel_uv`` is always ``(u, v) == (x, y)``.  ``source_image_size`` is
    always ``(width, height)``.  For native InternNav use
    ``coordinate_frame='front_down'`` and ``view_id='front_down'``.  For the
    structured panoramic System-2 use ``coordinate_frame='panoramic'`` and a
    horizontal ``view_id``.
    """

    pixel_uv: tuple[float, float]
    source_image_size: tuple[int, int]
    coordinate_frame: str
    view_id: str
    distance_m: float
    confidence: float
    camera_fx_px: float
    pixel_goal_source: str
    distance_source: str
    confidence_source: str
    system2_call_id: str | None = None


@dataclass(frozen=True)
class FutureHeatmapRender:
    """Rendered heatmaps and their complete provenance."""

    heatmaps: np.ndarray
    valid: bool
    reason: str | None
    coordinate_frame: str
    channel_order: tuple[str, ...]
    active_view: str | None
    center_uv_heatmap: tuple[float, float] | None
    sigma_px: float | None
    peak_value: float | None
    distance_m: float | None
    confidence: float | None
    provenance: dict[str, Any]

    def metadata(self) -> dict[str, Any]:
        """Return JSON-safe metadata without embedding the raster values."""

        return {
            "schema": FUTURE_HEATMAP_SCHEMA_VERSION,
            "valid": bool(self.valid),
            "reason": self.reason,
            "coordinate_frame": self.coordinate_frame,
            "channel_order": list(self.channel_order),
            "active_view": self.active_view,
            "heatmap_shape": [int(value) for value in self.heatmaps.shape],
            "heatmap_dtype": str(self.heatmaps.dtype),
            "heatmap_range": [0.0, 1.0],
            "center_uv_heatmap": (
                [float(value) for value in self.center_uv_heatmap]
                if self.center_uv_heatmap is not None
                else None
            ),
            "sigma_px": None if self.sigma_px is None else float(self.sigma_px),
            "peak_value": (
                None if self.peak_value is None else float(self.peak_value)
            ),
            "distance_m": (
                None if self.distance_m is None else float(self.distance_m)
            ),
            "confidence": (
                None if self.confidence is None else float(self.confidence)
            ),
            "brightness_semantics": "peak_equals_system2_confidence",
            "size_semantics": "near_larger_far_smaller_from_metric_depth",
            "center_semantics": "system2_pixel_goal_uv",
            "visualization_vmin": 0.0,
            "visualization_vmax": 1.0,
            "blob_name": FUTURE_HEATMAP_BLOB_NAME,
            "blob_mime_type": FUTURE_HEATMAP_BLOB_MIME,
            "provenance": dict(self.provenance),
        }

    def to_npz_bytes(self) -> bytes:
        """Serialize the raster as a compact, pickle-free NPZ attachment."""

        stream = io.BytesIO()
        np.savez_compressed(
            stream,
            heatmaps=np.ascontiguousarray(self.heatmaps, dtype=np.float32),
        )
        return stream.getvalue()


def _require_finite(value: float, name: str) -> float:
    resolved = float(value)
    if not math.isfinite(resolved):
        raise FutureHeatmapContractError(f"{name} must be finite")
    return resolved


def metric_depth_at_pixel(
    depth_m: np.ndarray,
    pixel_uv: tuple[float, float],
    source_image_size: tuple[int, int],
    *,
    neighborhood_radius: int = 2,
) -> float:
    """Return robust metric depth at ``pixel_uv`` using a local median.

    Depth must already be expressed in meters and must be spatially aligned
    with the image in which System-2 selected the pixel.  Invalid/zero depth
    samples are ignored; an all-invalid neighbourhood fails closed.
    """

    depth = np.asarray(depth_m, dtype=np.float32)
    if depth.ndim == 3 and depth.shape[-1] == 1:
        depth = depth[..., 0]
    if depth.ndim != 2:
        raise FutureHeatmapContractError(
            f"depth_m must be [H,W] or [H,W,1], got {depth.shape}"
        )
    width, height = (int(source_image_size[0]), int(source_image_size[1]))
    if width <= 0 or height <= 0:
        raise FutureHeatmapContractError("source_image_size must be positive")
    if depth.shape != (height, width):
        raise FutureHeatmapContractError(
            "depth/image alignment mismatch: "
            f"depth={depth.shape} image={(height, width)}"
        )
    u = _require_finite(pixel_uv[0], "pixel u")
    v = _require_finite(pixel_uv[1], "pixel v")
    if not (0.0 <= u < width and 0.0 <= v < height):
        raise FutureHeatmapContractError(
            f"pixel_uv {(u, v)} is outside source image {(width, height)}"
        )
    radius = int(neighborhood_radius)
    if radius < 0:
        raise FutureHeatmapContractError("neighborhood_radius must be >= 0")
    x = int(round(u))
    y = int(round(v))
    x = min(max(x, 0), width - 1)
    y = min(max(y, 0), height - 1)
    x0, x1 = max(0, x - radius), min(width, x + radius + 1)
    y0, y1 = max(0, y - radius), min(height, y + radius + 1)
    patch = depth[y0:y1, x0:x1]
    valid = patch[np.isfinite(patch) & (patch > 0.0)]
    if valid.size == 0:
        raise FutureHeatmapContractError(
            "no positive finite metric depth around System-2 pixel"
        )
    distance = float(np.median(valid))
    if not math.isfinite(distance) or distance <= 0.0:
        raise FutureHeatmapContractError("resolved metric depth is invalid")
    return distance


def pinhole_intrinsics_from_hfov(
    image_size: tuple[int, int],
    *,
    hfov_degrees: float,
) -> np.ndarray:
    """Construct the same centered pinhole intrinsics used by InternNav."""

    width, height = (int(image_size[0]), int(image_size[1]))
    if width <= 0 or height <= 0:
        raise FutureHeatmapContractError("image_size must be positive")
    hfov = _require_finite(hfov_degrees, "hfov_degrees")
    if not 0.0 < hfov < 180.0:
        raise FutureHeatmapContractError("hfov_degrees must be in (0,180)")
    focal = (width / 2.0) / math.tan(math.radians(hfov / 2.0))
    return np.array(
        [
            [focal, 0.0, (width - 1.0) / 2.0],
            [0.0, focal, (height - 1.0) / 2.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def project_lookdown_pixel_to_front(
    *,
    pixel_uv_lookdown: tuple[float, float],
    z_depth_lookdown_m: float,
    lookdown_intrinsics: np.ndarray,
    front_intrinsics: np.ndarray,
    front_image_size: tuple[int, int],
    lookdown_pitch_degrees: float = 30.0,
    agent_height_m: float = 1.25,
    flat_elevation_tolerance_m: float = 0.20,
) -> LookdownToFrontProjection:
    """Project a native look-down goal into the normal horizontal front view.

    Camera coordinates are OpenCV-style ``(+x right, +y down, +z forward)``.
    Native InternNav captures the look-down image after two 15-degree LOOKDOWN
    actions, so its optical axis is pitched down by +30 degrees relative to the
    horizontal front camera.  A look-down 3D point is transformed to front via
    the exact x-axis rotation ``R_x(+pitch)`` and then reprojected.

    The selected RGB/depth pixel lies on a floor/stair surface, whereas the
    historical heatmap represents camera centres.  After recovering that
    surface point, this function lifts it by the fixed InternNav agent-camera
    height (1.25 m).  Small estimated floor-height errors are snapped to zero
    relative elevation, so flat-floor targets lie exactly on the horizontal
    image horizon.  Larger elevation changes are retained, preserving upstairs
    and downstairs offsets.

    This function never clamps a point into view.  A point below/above the
    horizontal image returns ``valid=False`` so downstream code can leave all
    four heatmap channels empty or choose an explicit fallback.
    """

    source_u = _require_finite(pixel_uv_lookdown[0], "lookdown pixel u")
    source_v = _require_finite(pixel_uv_lookdown[1], "lookdown pixel v")
    source_depth = _require_finite(z_depth_lookdown_m, "z_depth_lookdown_m")
    if source_depth <= 0.0:
        raise FutureHeatmapContractError("z_depth_lookdown_m must be > 0")
    pitch_degrees = _require_finite(
        lookdown_pitch_degrees, "lookdown_pitch_degrees"
    )
    if not 0.0 <= pitch_degrees < 90.0:
        raise FutureHeatmapContractError(
            "lookdown_pitch_degrees must be in [0,90)"
        )
    resolved_agent_height = _require_finite(agent_height_m, "agent_height_m")
    if resolved_agent_height <= 0.0:
        raise FutureHeatmapContractError("agent_height_m must be > 0")
    flat_tolerance = _require_finite(
        flat_elevation_tolerance_m, "flat_elevation_tolerance_m"
    )
    if flat_tolerance < 0.0:
        raise FutureHeatmapContractError(
            "flat_elevation_tolerance_m must be >= 0"
        )
    lookdown_k = np.asarray(lookdown_intrinsics, dtype=np.float64)
    front_k = np.asarray(front_intrinsics, dtype=np.float64)
    if lookdown_k.shape != (3, 3) or front_k.shape != (3, 3):
        raise FutureHeatmapContractError(
            "lookdown_intrinsics and front_intrinsics must both be [3,3]"
        )
    if not np.isfinite(lookdown_k).all() or not np.isfinite(front_k).all():
        raise FutureHeatmapContractError("camera intrinsics must be finite")
    for matrix_name, matrix in (
        ("lookdown_intrinsics", lookdown_k),
        ("front_intrinsics", front_k),
    ):
        if matrix[0, 0] <= 0.0 or matrix[1, 1] <= 0.0:
            raise FutureHeatmapContractError(
                f"{matrix_name} focal lengths must be positive"
            )
    width, height = (int(front_image_size[0]), int(front_image_size[1]))
    if width <= 0 or height <= 0:
        raise FutureHeatmapContractError("front_image_size must be positive")

    # Backproject using z-depth, matching InternNav's pixel_to_gps semantics.
    point_lookdown = np.array(
        [
            (source_u - lookdown_k[0, 2])
            * source_depth
            / lookdown_k[0, 0],
            (source_v - lookdown_k[1, 2])
            * source_depth
            / lookdown_k[1, 1],
            source_depth,
        ],
        dtype=np.float64,
    )
    theta = math.radians(pitch_degrees)
    cosine = math.cos(theta)
    sine = math.sin(theta)
    lookdown_to_front = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, cosine, sine],
            [0.0, -sine, cosine],
        ],
        dtype=np.float64,
    )
    surface_point_front = lookdown_to_front @ point_lookdown
    surface_x, surface_y, surface_z = (
        float(value) for value in surface_point_front
    )
    # In OpenCV coordinates +y points down.  A flat floor is therefore at
    # y=+agent_height below the current camera.  The raw elevation is positive
    # for an upstairs target and negative for a downstairs target.
    raw_elevation_delta = resolved_agent_height - surface_y
    if abs(raw_elevation_delta) <= flat_tolerance:
        used_elevation_delta = 0.0
        height_mode = "flat_agent_height_snapped"
    else:
        used_elevation_delta = raw_elevation_delta
        height_mode = "height_change_preserved"
    point_front = np.array(
        [surface_x, -used_elevation_delta, surface_z], dtype=np.float64
    )
    x_front, y_front, z_front = (float(value) for value in point_front)
    common = {
        "source_pixel_uv": (float(source_u), float(source_v)),
        "source_depth_m": float(source_depth),
        "lookdown_pitch_degrees": float(pitch_degrees),
        "agent_height_m": float(resolved_agent_height),
        "raw_elevation_delta_m": float(raw_elevation_delta),
        "used_elevation_delta_m": float(used_elevation_delta),
        "flat_elevation_tolerance_m": float(flat_tolerance),
        "height_mode": height_mode,
        "surface_point_front_xyz_m": (
            float(surface_x),
            float(surface_y),
            float(surface_z),
        ),
    }
    if not math.isfinite(z_front) or z_front <= 1e-4:
        return LookdownToFrontProjection(
            valid=False,
            reason="projected_point_is_behind_horizontal_front_camera",
            pixel_uv_front=None,
            point_front_xyz_m=(x_front, y_front, z_front),
            z_depth_front_m=None,
            **common,
        )
    front_u = front_k[0, 0] * x_front / z_front + front_k[0, 2]
    front_v = front_k[1, 1] * y_front / z_front + front_k[1, 2]
    if not (
        math.isfinite(front_u)
        and math.isfinite(front_v)
        and 0.0 <= front_u < width
        and 0.0 <= front_v < height
    ):
        return LookdownToFrontProjection(
            valid=False,
            reason="projected_point_is_outside_horizontal_front_view",
            pixel_uv_front=(float(front_u), float(front_v)),
            point_front_xyz_m=(x_front, y_front, z_front),
            z_depth_front_m=float(z_front),
            **common,
        )
    return LookdownToFrontProjection(
        valid=True,
        reason=None,
        pixel_uv_front=(float(front_u), float(front_v)),
        point_front_xyz_m=(x_front, y_front, z_front),
        z_depth_front_m=float(z_front),
        **common,
    )


def geometric_mean_probability(log_probabilities: np.ndarray | list[float]) -> float:
    """Convert token log-probabilities into an uncalibrated sequence score."""

    values = np.asarray(log_probabilities, dtype=np.float64).reshape(-1)
    if values.size == 0 or not np.isfinite(values).all():
        raise FutureHeatmapContractError(
            "token log-probabilities must be a non-empty finite sequence"
        )
    probability = float(np.exp(values.mean()))
    return float(np.clip(probability, 0.0, 1.0))


class FutureHeatmapRenderer:
    """No-parameter renderer matching the existing Gaussian heatmap style."""

    def __init__(
        self,
        *,
        heatmap_size: tuple[int, int] = (64, 64),
        object_size_m: float = 1.5,
        min_sigma_px: float = 4.0,
        max_sigma_px: float = 8.0,
    ) -> None:
        self.heatmap_size = (int(heatmap_size[0]), int(heatmap_size[1]))
        self.object_size_m = _require_finite(object_size_m, "object_size_m")
        self.min_sigma_px = _require_finite(min_sigma_px, "min_sigma_px")
        self.max_sigma_px = _require_finite(max_sigma_px, "max_sigma_px")
        if min(self.heatmap_size) <= 0:
            raise FutureHeatmapContractError("heatmap_size must be positive")
        if self.object_size_m <= 0:
            raise FutureHeatmapContractError("object_size_m must be > 0")
        if self.min_sigma_px <= 0 or self.max_sigma_px < self.min_sigma_px:
            raise FutureHeatmapContractError("invalid sigma bounds")

    @staticmethod
    def channel_order_for_frame(coordinate_frame: str) -> tuple[str, ...]:
        if coordinate_frame == FRONT_DOWN_VIEW:
            return (FRONT_DOWN_VIEW,)
        if coordinate_frame == "panoramic":
            return PANORAMIC_VIEW_ORDER
        raise FutureHeatmapContractError(
            "coordinate_frame must be 'front_down' or 'panoramic'"
        )

    def invalid(
        self,
        *,
        coordinate_frame: str,
        reason: str,
        provenance: dict[str, Any] | None = None,
    ) -> FutureHeatmapRender:
        order = self.channel_order_for_frame(coordinate_frame)
        hm_width, hm_height = self.heatmap_size
        return FutureHeatmapRender(
            heatmaps=np.zeros(
                (len(order), hm_height, hm_width), dtype=np.float32
            ),
            valid=False,
            reason=str(reason),
            coordinate_frame=coordinate_frame,
            channel_order=order,
            active_view=None,
            center_uv_heatmap=None,
            sigma_px=None,
            peak_value=None,
            distance_m=None,
            confidence=None,
            provenance=dict(provenance or {}),
        )

    def render(self, evidence: FutureGoalEvidence) -> FutureHeatmapRender:
        order = self.channel_order_for_frame(evidence.coordinate_frame)
        if evidence.coordinate_frame == FRONT_DOWN_VIEW:
            if evidence.view_id != FRONT_DOWN_VIEW:
                raise FutureHeatmapContractError(
                    "front_down evidence must use view_id='front_down'"
                )
        elif evidence.view_id not in PANORAMIC_VIEW_ORDER:
            raise FutureHeatmapContractError(
                f"panoramic view_id must be one of {PANORAMIC_VIEW_ORDER}"
            )

        width, height = (
            int(evidence.source_image_size[0]),
            int(evidence.source_image_size[1]),
        )
        if width <= 0 or height <= 0:
            raise FutureHeatmapContractError("source_image_size must be positive")
        u = _require_finite(evidence.pixel_uv[0], "pixel u")
        v = _require_finite(evidence.pixel_uv[1], "pixel v")
        if not (0.0 <= u < width and 0.0 <= v < height):
            raise FutureHeatmapContractError(
                f"pixel_uv {(u, v)} is outside source image {(width, height)}"
            )
        distance = _require_finite(evidence.distance_m, "distance_m")
        if distance <= 0.0:
            raise FutureHeatmapContractError("distance_m must be > 0")
        confidence = _require_finite(evidence.confidence, "confidence")
        if not 0.0 <= confidence <= 1.0:
            raise FutureHeatmapContractError("confidence must be in [0,1]")
        camera_fx = _require_finite(evidence.camera_fx_px, "camera_fx_px")
        if camera_fx <= 0.0:
            raise FutureHeatmapContractError("camera_fx_px must be > 0")
        for name, value in (
            ("pixel_goal_source", evidence.pixel_goal_source),
            ("distance_source", evidence.distance_source),
            ("confidence_source", evidence.confidence_source),
        ):
            if not str(value).strip():
                raise FutureHeatmapContractError(f"{name} must be explicit")

        hm_width, hm_height = self.heatmap_size
        u_hm = u * hm_width / width
        v_hm = v * hm_height / height
        projected_size_hm = (
            self.object_size_m * camera_fx / distance * hm_width / width
        )
        sigma = float(
            np.clip(
                projected_size_hm / 3.0,
                self.min_sigma_px,
                self.max_sigma_px,
            )
        )

        grid_y, grid_x = np.mgrid[0:hm_height, 0:hm_width].astype(np.float32)
        gaussian = np.exp(
            -((grid_x - u_hm) ** 2 + (grid_y - v_hm) ** 2)
            / (2.0 * sigma**2)
        ).astype(np.float32)
        # The old renderer truncates at 3 sigma.  Match that visual support,
        # then normalize the discrete raster so its actual peak is exactly c.
        support = (
            (grid_x - u_hm) ** 2 + (grid_y - v_hm) ** 2
        ) <= (3.0 * sigma) ** 2
        gaussian *= support.astype(np.float32)
        max_value = float(gaussian.max(initial=0.0))
        if max_value <= 0.0:
            raise FutureHeatmapContractError("Gaussian has no in-frame support")
        gaussian *= np.float32(confidence / max_value)

        heatmaps = np.zeros(
            (len(order), hm_height, hm_width), dtype=np.float32
        )
        heatmaps[order.index(evidence.view_id)] = gaussian
        provenance = {
            "pixel_goal_source": evidence.pixel_goal_source,
            "distance_source": evidence.distance_source,
            "confidence_source": evidence.confidence_source,
            "source_image_size": [width, height],
            "source_pixel_uv": [u, v],
            "camera_fx_px": camera_fx,
            "object_size_m": self.object_size_m,
            "sigma_clip_px": [self.min_sigma_px, self.max_sigma_px],
            "system2_call_id": evidence.system2_call_id,
        }
        return FutureHeatmapRender(
            heatmaps=np.ascontiguousarray(heatmaps),
            valid=True,
            reason=None,
            coordinate_frame=evidence.coordinate_frame,
            channel_order=order,
            active_view=evidence.view_id,
            center_uv_heatmap=(float(u_hm), float(v_hm)),
            sigma_px=sigma,
            peak_value=float(heatmaps.max(initial=0.0)),
            distance_m=distance,
            confidence=confidence,
            provenance=provenance,
        )

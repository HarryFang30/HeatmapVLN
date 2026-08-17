from __future__ import annotations

import io

import numpy as np
import pytest

from src.models.heatmap.future_heatmap_renderer import (
    FRONT_DOWN_VIEW,
    FUTURE_HEATMAP_SCHEMA_VERSION,
    FutureGoalEvidence,
    FutureHeatmapContractError,
    FutureHeatmapRenderer,
    geometric_mean_probability,
    metric_depth_at_pixel,
    pinhole_intrinsics_from_hfov,
    project_lookdown_pixel_to_front,
)


def _evidence(**overrides) -> FutureGoalEvidence:
    values = {
        "pixel_uv": (320.0, 240.0),
        "source_image_size": (640, 480),
        "coordinate_frame": FRONT_DOWN_VIEW,
        "view_id": FRONT_DOWN_VIEW,
        "distance_m": 3.0,
        "confidence": 0.73,
        "camera_fx_px": 388.0,
        "pixel_goal_source": "internnav_native_system2_uv",
        "distance_source": "sensor_depth_at_system2_pixel",
        "confidence_source": (
            "system2_sequence_geomean_token_probability_uncalibrated"
        ),
        "system2_call_id": "scene/episode/call-0",
    }
    values.update(overrides)
    return FutureGoalEvidence(**values)


def test_native_front_down_render_has_old_heatmap_shape_and_exact_peak() -> None:
    renderer = FutureHeatmapRenderer()
    result = renderer.render(_evidence())

    assert result.valid
    assert result.heatmaps.shape == (1, 64, 64)
    assert result.heatmaps.dtype == np.float32
    assert result.channel_order == (FRONT_DOWN_VIEW,)
    assert result.center_uv_heatmap == pytest.approx((32.0, 32.0))
    assert float(result.heatmaps.max()) == pytest.approx(0.73, abs=1e-6)
    assert result.peak_value == pytest.approx(0.73, abs=1e-6)
    metadata = result.metadata()
    assert metadata["schema"] == FUTURE_HEATMAP_SCHEMA_VERSION
    assert metadata["visualization_vmin"] == 0.0
    assert metadata["visualization_vmax"] == 1.0
    assert metadata["brightness_semantics"] == "peak_equals_system2_confidence"


def test_panoramic_render_activates_only_explicit_view() -> None:
    renderer = FutureHeatmapRenderer()
    result = renderer.render(
        _evidence(
            coordinate_frame="panoramic",
            view_id="left",
            source_image_size=(384, 384),
            pixel_uv=(96.0, 192.0),
            camera_fx_px=192.0,
        )
    )

    assert result.heatmaps.shape == (4, 64, 64)
    assert result.channel_order == ("front", "right", "back", "left")
    assert np.count_nonzero(result.heatmaps[:3]) == 0
    assert np.count_nonzero(result.heatmaps[3]) > 0
    assert result.center_uv_heatmap == pytest.approx((16.0, 32.0))


def test_distance_changes_size_but_not_brightness() -> None:
    renderer = FutureHeatmapRenderer()
    near = renderer.render(_evidence(distance_m=2.0, confidence=0.61))
    far = renderer.render(_evidence(distance_m=6.0, confidence=0.61))

    assert near.sigma_px > far.sigma_px
    assert near.peak_value == pytest.approx(0.61, abs=1e-6)
    assert far.peak_value == pytest.approx(0.61, abs=1e-6)
    near_area = int((near.heatmaps >= 0.5 * 0.61).sum())
    far_area = int((far.heatmaps >= 0.5 * 0.61).sum())
    assert near_area > far_area


def test_confidence_changes_brightness_but_not_size() -> None:
    renderer = FutureHeatmapRenderer()
    low = renderer.render(_evidence(confidence=0.2))
    high = renderer.render(_evidence(confidence=0.9))

    assert low.sigma_px == pytest.approx(high.sigma_px)
    assert low.peak_value == pytest.approx(0.2, abs=1e-6)
    assert high.peak_value == pytest.approx(0.9, abs=1e-6)
    # Once divided by their explicit confidence, the shapes are identical.
    np.testing.assert_allclose(low.heatmaps / 0.2, high.heatmaps / 0.9)


def test_metric_depth_uses_aligned_uv_neighbourhood_median() -> None:
    depth = np.zeros((480, 640), dtype=np.float32)
    depth[238:243, 318:323] = 4.0
    depth[240, 320] = np.nan
    assert metric_depth_at_pixel(depth, (320, 240), (640, 480)) == 4.0


@pytest.mark.parametrize(
    "kwargs, message",
    [
        ({"pixel_uv": (640.0, 10.0)}, "outside source image"),
        ({"distance_m": 0.0}, "distance_m must be > 0"),
        ({"confidence": 1.1}, "confidence must be in"),
        ({"confidence_source": ""}, "confidence_source must be explicit"),
        ({"view_id": "front"}, "front_down evidence"),
    ],
)
def test_invalid_evidence_fails_closed(kwargs, message) -> None:
    with pytest.raises(FutureHeatmapContractError, match=message):
        FutureHeatmapRenderer().render(_evidence(**kwargs))


def test_invalid_stop_or_missing_evidence_is_zero() -> None:
    result = FutureHeatmapRenderer().invalid(
        coordinate_frame=FRONT_DOWN_VIEW,
        reason="system2_stop_has_no_future_point",
        provenance={"pixel_goal_source": "none"},
    )
    assert not result.valid
    assert result.reason == "system2_stop_has_no_future_point"
    assert result.heatmaps.shape == (1, 64, 64)
    assert np.count_nonzero(result.heatmaps) == 0


def test_geometric_mean_probability_and_pickle_free_npz() -> None:
    confidence = geometric_mean_probability(np.log([0.25, 1.0]))
    assert confidence == pytest.approx(0.5)

    result = FutureHeatmapRenderer().render(_evidence(confidence=confidence))
    payload = result.to_npz_bytes()
    with np.load(io.BytesIO(payload), allow_pickle=False) as archive:
        assert archive.files == ["heatmaps"]
        np.testing.assert_array_equal(archive["heatmaps"], result.heatmaps)


def test_depth_alignment_and_empty_patch_fail_closed() -> None:
    with pytest.raises(FutureHeatmapContractError, match="alignment mismatch"):
        metric_depth_at_pixel(
            np.ones((64, 64), dtype=np.float32),
            (10, 10),
            (640, 480),
        )
    with pytest.raises(FutureHeatmapContractError, match="no positive finite"):
        metric_depth_at_pixel(
            np.zeros((480, 640), dtype=np.float32),
            (10, 10),
            (640, 480),
        )


def test_flat_floor_target_is_lifted_to_agent_height_and_horizon() -> None:
    size = (640, 480)
    intrinsics = pinhole_intrinsics_from_hfov(size, hfov_degrees=90.0)
    depth = 3.0
    pitch = np.deg2rad(30.0)
    # Select the lookdown ray whose rotated surface point is exactly 1.25m
    # below the current camera (a flat floor).
    y_lookdown = (1.25 - np.sin(pitch) * depth) / np.cos(pitch)
    v_lookdown = intrinsics[1, 2] + intrinsics[1, 1] * y_lookdown / depth
    result = project_lookdown_pixel_to_front(
        pixel_uv_lookdown=(intrinsics[0, 2], v_lookdown),
        z_depth_lookdown_m=depth,
        lookdown_intrinsics=intrinsics,
        front_intrinsics=intrinsics,
        front_image_size=size,
        lookdown_pitch_degrees=30.0,
    )
    assert result.valid
    assert result.pixel_uv_front is not None
    assert result.pixel_uv_front[0] == pytest.approx(intrinsics[0, 2])
    assert result.pixel_uv_front[1] == pytest.approx(intrinsics[1, 2])
    assert result.raw_elevation_delta_m == pytest.approx(0.0, abs=1e-6)
    assert result.used_elevation_delta_m == 0.0
    assert result.height_mode == "flat_agent_height_snapped"
    assert result.point_front_xyz_m is not None
    assert result.point_front_xyz_m[1] == 0.0


def test_large_floor_elevation_change_is_preserved_for_stairs() -> None:
    size = (640, 480)
    intrinsics = pinhole_intrinsics_from_hfov(size, hfov_degrees=90.0)
    # The centre lookdown ray at 3m hits a surface 1.5m below the camera.
    # Relative to a 1.25m agent height this is a 0.25m downward transition,
    # beyond the 0.20m flat snap threshold, so it must remain visible below
    # the horizon after lifting the target to future agent-camera height.
    result = project_lookdown_pixel_to_front(
        pixel_uv_lookdown=(intrinsics[0, 2], intrinsics[1, 2]),
        z_depth_lookdown_m=3.0,
        lookdown_intrinsics=intrinsics,
        front_intrinsics=intrinsics,
        front_image_size=size,
        lookdown_pitch_degrees=30.0,
    )
    assert result.valid
    assert result.height_mode == "height_change_preserved"
    assert result.raw_elevation_delta_m == pytest.approx(-0.25)
    assert result.used_elevation_delta_m == pytest.approx(-0.25)
    assert result.point_front_xyz_m is not None
    assert result.point_front_xyz_m[1] == pytest.approx(0.25)
    assert result.pixel_uv_front is not None
    assert result.pixel_uv_front[1] > intrinsics[1, 2]


def test_out_of_front_view_projection_is_not_clamped() -> None:
    size = (640, 480)
    intrinsics = pinhole_intrinsics_from_hfov(size, hfov_degrees=90.0)
    result = project_lookdown_pixel_to_front(
        pixel_uv_lookdown=(320.0, 470.0),
        z_depth_lookdown_m=2.0,
        lookdown_intrinsics=intrinsics,
        front_intrinsics=intrinsics,
        front_image_size=size,
        lookdown_pitch_degrees=30.0,
    )
    assert not result.valid
    assert result.reason == "projected_point_is_outside_horizontal_front_view"
    assert result.pixel_uv_front is not None
    assert result.pixel_uv_front[1] >= size[1]


def test_projection_rejects_nonphysical_depth_and_intrinsics() -> None:
    intrinsics = pinhole_intrinsics_from_hfov((640, 480), hfov_degrees=90.0)
    with pytest.raises(FutureHeatmapContractError, match="must be > 0"):
        project_lookdown_pixel_to_front(
            pixel_uv_lookdown=(320.0, 240.0),
            z_depth_lookdown_m=0.0,
            lookdown_intrinsics=intrinsics,
            front_intrinsics=intrinsics,
            front_image_size=(640, 480),
        )
    with pytest.raises(FutureHeatmapContractError, match="agent_height_m"):
        project_lookdown_pixel_to_front(
            pixel_uv_lookdown=(320.0, 240.0),
            z_depth_lookdown_m=1.0,
            lookdown_intrinsics=intrinsics,
            front_intrinsics=intrinsics,
            front_image_size=(640, 480),
            agent_height_m=0.0,
        )
    with pytest.raises(FutureHeatmapContractError, match=r"must both be \[3,3\]"):
        project_lookdown_pixel_to_front(
            pixel_uv_lookdown=(320.0, 240.0),
            z_depth_lookdown_m=1.0,
            lookdown_intrinsics=np.eye(4),
            front_intrinsics=intrinsics,
            front_image_size=(640, 480),
        )

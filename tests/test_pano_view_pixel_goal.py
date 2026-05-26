import json

from src.data.pano_view_pixel_goal import (
    VisibleProjection,
    load_intrinsics,
    select_canonical_view,
)


def test_select_canonical_view_prefers_depth_checked_front():
    visible = [
        VisibleProjection("front", 250, 128, 122.0, 2.0),
        VisibleProjection("left", 128, 128, 0.0, 2.0),
    ]

    assert select_canonical_view(visible).view_id == "front"


def test_select_canonical_view_uses_center_distance_without_front():
    visible = [
        VisibleProjection("right", 240, 128, 112.0, 2.0),
        VisibleProjection("back", 130, 128, 2.0, 2.0),
        VisibleProjection("left", 40, 128, 88.0, 2.0),
    ]

    chosen = select_canonical_view(visible)
    assert chosen.view_id == "back"
    assert chosen.u == 130


def test_load_intrinsics_accepts_k_matrix(tmp_path):
    (tmp_path / "intrinsics.json").write_text(json.dumps({
        "width": 256,
        "height": 256,
        "K": [
            [128.0, 0.0, 127.5],
            [0.0, 128.0, 127.5],
            [0.0, 0.0, 1.0],
        ],
    }))

    intrinsics = load_intrinsics(tmp_path)
    assert intrinsics == {
        "width": 256.0,
        "height": 256.0,
        "fx": 128.0,
        "fy": 128.0,
        "cx": 127.5,
        "cy": 127.5,
    }

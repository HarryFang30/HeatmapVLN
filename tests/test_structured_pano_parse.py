from src.models.heatmap.input_constructor import (
    parse_structured_pano_output,
    structured_condition_text,
    vlm_output_requests_stop,
)


def test_parse_structured_pano_pixel_goal():
    parsed = parse_structured_pano_output(
        "view: right\npixel: 211 128",
        image_size=(256, 256),
    )
    assert parsed.kind == "pixel"
    assert parsed.view_id == "right"
    assert parsed.pixel_goal == [211, 128]


def test_parse_structured_pano_stop_and_turn():
    assert parse_structured_pano_output("view: stop", image_size=None).kind == "stop"
    assert parse_structured_pano_output("view: turn", image_size=None).kind == "turn"
    assert vlm_output_requests_stop("view: stop")
    assert not vlm_output_requests_stop("view: turn")


def test_parse_legacy_coord_fallback():
    parsed = parse_structured_pano_output("128 192", image_size=(256, 256))
    assert parsed.kind == "legacy_coord"
    assert parsed.view_id == "front"
    assert parsed.pixel_goal == [128, 192]


def test_structured_condition_text():
    text = structured_condition_text("right", [211, 128])
    assert text == "view: right\npixel: 211 128"


def test_parse_clamps_out_of_bounds_pixel():
    parsed = parse_structured_pano_output(
        "view: front\npixel: 999 999",
        image_size=(256, 256),
    )
    assert parsed.pixel_goal == [255, 255]

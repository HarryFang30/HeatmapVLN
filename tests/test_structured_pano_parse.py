from src.models.heatmap.input_constructor import (
    parse_structured_pano_output,
    structured_condition_text,
    vlm_output_requests_stop,
    vlm_output_requests_turn,
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
    assert parse_structured_pano_output("view: turn", image_size=None).turn_direction is None
    assert vlm_output_requests_stop("view: stop")
    assert not vlm_output_requests_stop("view: turn")


def test_parse_structured_pano_directional_turns():
    left = parse_structured_pano_output("view: turn_left", image_size=None)
    assert left.kind == "turn"
    assert left.turn_direction == "left"

    right = parse_structured_pano_output("view: turn_right", image_size=None)
    assert right.kind == "turn"
    assert right.turn_direction == "right"

    assert vlm_output_requests_turn("view: turn_left") == "left"
    assert vlm_output_requests_turn("view: turn_right") == "right"
    assert vlm_output_requests_turn("view: turn") is None  # ambiguous
    assert vlm_output_requests_turn("view: front\npixel: 100 200") is None


def test_parse_legacy_coord_fallback():
    parsed = parse_structured_pano_output("128 192", image_size=(256, 256))
    assert parsed.kind == "legacy_coord"
    assert parsed.view_id == "front"
    assert parsed.pixel_goal == [128, 192]


def test_parse_structured_pano_inline_pixel_goal():
    parsed = parse_structured_pano_output(
        "view: right pixel: 211 128",
        image_size=(256, 256),
    )
    assert parsed.kind == "pixel"
    assert parsed.view_id == "right"
    assert parsed.pixel_goal == [211, 128]


def test_parse_pixel_only_legacy_front_fallback():
    parsed = parse_structured_pano_output(
        "pixel: 211 128",
        image_size=(256, 256),
    )
    assert parsed.kind == "legacy_coord"
    assert parsed.view_id == "front"
    assert parsed.pixel_goal == [211, 128]


def test_structured_option_echo_is_invalid_not_front():
    parsed = parse_structured_pano_output(
        "view: front|right|back|left\npixel: 10 128",
        image_size=(256, 256),
    )
    assert parsed.kind == "invalid"
    assert parsed.view_id is None


def test_malformed_structured_output_does_not_use_legacy_fallback():
    parsed = parse_structured_pano_output(
        "view: right|back\npixel: 10 128",
        image_size=(256, 256),
    )
    assert parsed.kind == "invalid"


def test_xml_box_output_is_not_legacy_coord():
    parsed = parse_structured_pano_output(
        "<ref>front</ref><box>[[190,164,231,211]]</box>",
        image_size=(256, 256),
    )
    assert parsed.kind == "invalid"


def test_legacy_coord_can_be_disabled_for_structured_eval():
    parsed = parse_structured_pano_output(
        "128 192",
        image_size=(256, 256),
        allow_legacy_coord=False,
    )
    assert parsed.kind == "invalid"


def test_structured_condition_text():
    text = structured_condition_text("right", [211, 128])
    assert text == "view: right\npixel: 211 128"


def test_parse_clamps_out_of_bounds_pixel():
    parsed = parse_structured_pano_output(
        "view: front\npixel: 999 999",
        image_size=(256, 256),
    )
    assert parsed.pixel_goal == [255, 255]

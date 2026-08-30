"""Golden contracts for the byte-exact native InternNav System2 front end."""

from __future__ import annotations

from PIL import Image

from src.models.heatmap.native_internnav_exact import (
    NATIVE_LOOKDOWN_SIZE,
    append_native_lookdown_turn,
    build_native_messages,
    finalize_native_local_actions,
    native_requests_lookdown,
    parse_native_actions,
)


def _img(size=(384, 384)) -> Image.Image:
    return Image.new("RGB", size, color=(7, 7, 7))


def _texts(message: dict) -> list[str]:
    return [
        part["text"] for part in message["content"] if part["type"] == "text"
    ]


def _layout(message: dict) -> list[str]:
    return [part["type"] for part in message["content"]]


BASE_WITH_PERIOD_INSTRUCTION = (
    "You are an autonomous navigation assistant. Your task is to "
    "Walk to the couch. Where should you go next to stay on track? "
    "Please output the next waypoint's coordinates in the image. "
    "Please output STOP when you have successfully completed the task."
)


class TestNativePromptGolden:
    def test_no_history_prompt_matches_replica_rendering(self) -> None:
        messages, images = build_native_messages(
            "Walk to the couch.", [], _img()
        )
        assert len(messages) == 1 and messages[0]["role"] == "user"
        assert _layout(messages[0]) == ["text", "image", "text"]
        assert _texts(messages[0]) == [
            f"{BASE_WITH_PERIOD_INSTRUCTION} you can see",
            ".",
        ]
        assert len(images) == 1

    def test_instruction_period_is_never_doubled(self) -> None:
        # The replica replaces "<instruction>." wholesale: an instruction that
        # already ends with a period yields exactly one period, and one
        # without a period yields none (official behavior, locked here).
        with_period, _ = build_native_messages("Go left.", [], _img())
        assert "Go left. Where should" in _texts(with_period[0])[0]
        assert "Go left.. " not in _texts(with_period[0])[0]
        without_period, _ = build_native_messages("Go left", [], _img())
        assert "Go left Where should" in _texts(without_period[0])[0]

    def test_history_images_have_no_separator_text(self) -> None:
        history = [_img(), _img()]
        messages, images = build_native_messages(
            "Walk to the couch.", history, _img()
        )
        # Replica cleaning drops the "\n" between history images entirely:
        # [prefix text][img][img][". you can see"][img]["."]
        assert _layout(messages[0]) == [
            "text",
            "image",
            "image",
            "text",
            "image",
            "text",
        ]
        texts = _texts(messages[0])
        assert texts[0].endswith("These are your historical observations:")
        assert texts[1] == ". you can see"
        assert texts[2] == "."
        assert len(images) == 3

    def test_lookdown_turn_appends_assistant_and_user(self) -> None:
        messages, images = build_native_messages(
            "Walk to the couch.", [], _img()
        )
        lookdown = _img(NATIVE_LOOKDOWN_SIZE)
        extended, extended_images = append_native_lookdown_turn(
            messages, images, "↓", lookdown
        )
        assert len(messages) == 1  # original untouched (deepcopy)
        assert [m["role"] for m in extended] == ["user", "assistant", "user"]
        assert _texts(extended[1]) == ["↓"]
        assert _layout(extended[2]) == ["text", "image", "text"]
        assert _texts(extended[2]) == ["you can see", "."]
        assert len(extended_images) == 2
        assert extended_images[-1].size == NATIVE_LOOKDOWN_SIZE


class TestNativeActionSemantics:
    def test_parse_covers_all_official_tokens_in_order(self) -> None:
        assert parse_native_actions("←←→STOP↑↓") == [2, 2, 3, 0, 1, 5]
        assert parse_native_actions("") == []
        assert parse_native_actions(None) == []

    def test_finalize_pads_with_stop_then_caps_to_four(self) -> None:
        assert finalize_native_local_actions([3, 3]) == [3, 3, 0, 0]
        assert finalize_native_local_actions([2, 2, 2, 2, 2, 2]) == [2, 2, 2, 2]
        assert finalize_native_local_actions([]) == [0, 0, 0, 0]

    def test_lookdown_request_requires_leading_arrow_and_no_digits(self) -> None:
        assert native_requests_lookdown("↓") is True
        assert native_requests_lookdown("↓↓") is True
        assert native_requests_lookdown("163 203") is False
        assert native_requests_lookdown("←↓") is False
        assert native_requests_lookdown("") is False
        assert native_requests_lookdown("↓ 12") is False

    def test_lookdown_size_matches_released_sensor(self) -> None:
        assert NATIVE_LOOKDOWN_SIZE == (640, 480)

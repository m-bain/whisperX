"""Test that highlighted subtitle cues are joined like the plain ones."""

from whisperx.utils import WriteSRT


def _cues(language, words):
    result = {
        "language": language,
        "segments": [
            {
                "start": 0.0,
                "end": 2.0,
                "words": [
                    {"word": w, "start": i * 0.5, "end": i * 0.5 + 0.5}
                    for i, w in enumerate(words)
                ],
            }
        ],
    }
    writer = WriteSRT(".")
    options = {"max_line_width": None, "max_line_count": None, "highlight_words": True}
    return [text for _, _, text in writer.iterate_result(result, options)]


def test_highlight_words_has_no_spaces_for_languages_without_spaces():
    assert _cues("zh", list("你好世界")) == [
        "<u>你</u>好世界",
        "你<u>好</u>世界",
        "你好<u>世</u>界",
        "你好世<u>界</u>",
    ]


def test_highlight_words_keeps_spaces_for_other_languages():
    assert _cues("en", ["Hello", "world"]) == ["<u>Hello</u> world", "Hello <u>world</u>"]

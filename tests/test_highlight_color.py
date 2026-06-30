"""Unit tests for the --highlight_color feature in subtitle output."""

import pytest

from whisperx.utils import (
    _apply_highlight,
    _parse_score_color_map,
    _select_score_color,
    WriteVTT,
    WriteSRT,
)


class TestApplyHighlight:
    """Unit tests for _apply_highlight."""

    def test_not_current_returns_word_unchanged(self):
        assert _apply_highlight("hello", None, is_current=False) == "hello"
        assert _apply_highlight("world", "red", is_current=False) == "world"
        assert _apply_highlight(" foo", None, is_current=False) == " foo"

    def test_current_no_color_adds_underline(self):
        assert _apply_highlight("hello", None, is_current=True) == "<u>hello</u>"

    def test_current_with_color_adds_font_tag(self):
        assert _apply_highlight("hello", "red", is_current=True) == '<font color="red">hello</font>'
        assert _apply_highlight("hello", "#00ff00", is_current=True) == '<font color="#00ff00">hello</font>'

    def test_current_preserves_leading_whitespace(self):
        assert _apply_highlight("\nhello", None, is_current=True) == "\n<u>hello</u>"
        assert _apply_highlight("\nworld", "red", is_current=True) == '\n<font color="red">world</font>'

    def test_tags_balanced_for_every_input(self):
        """Every call with is_current=True produces balanced open/close tags."""
        cases = [
            ("word", None),
            ("word", "red"),
            ("word", "#ff0000"),
            ("\nword", None),
            (" word", "green"),
        ]
        for word, color in cases:
            result = _apply_highlight(word, color, is_current=True)
            if color:
                assert result.count(f'<font color="{color}">') == result.count("</font>"), (
                    f"Unbalanced font tags in: {result!r}"
                )
            else:
                assert result.count("<u>") == result.count("</u>"), (
                    f"Unbalanced u tags in: {result!r}"
                )

    def test_score_color_map_overrides_fixed_color(self):
        score_color_map = [(0.5, "red"), (0.8, "green")]
        result = _apply_highlight("hello", "blue", is_current=True, score_color_map=score_color_map, score=0.9)
        assert '<font color="green">' in result

    def test_score_color_map_falls_back_when_score_missing(self):
        score_color_map = [(0.5, "red"), (0.8, "green")]
        result = _apply_highlight("hello", "blue", is_current=True, score_color_map=score_color_map, score=None)
        assert '<font color="blue">' in result

    def test_score_color_map_selects_color_by_threshold(self):
        score_color_map = [(0.3, "red"), (0.7, "yellow"), (1.0, "green")]
        low = _apply_highlight("x", None, is_current=True, score_color_map=score_color_map, score=0.2)
        mid = _apply_highlight("x", None, is_current=True, score_color_map=score_color_map, score=0.5)
        high = _apply_highlight("x", None, is_current=True, score_color_map=score_color_map, score=0.9)
        assert '<font color="red">' in low
        assert '<font color="yellow">' in mid
        assert '<font color="green">' in high


class TestScoreColorMapHelpers:
    """Unit tests for score color map parsing and selection."""

    def test_parse_score_color_map(self):
        assert _parse_score_color_map("0:red,0.5:yellow,0.8:green") == [
            (0.0, "red"),
            (0.5, "yellow"),
            (0.8, "green"),
        ]

    def test_parse_score_color_map_sorts_by_threshold(self):
        assert _parse_score_color_map("0.8:green,0:red,0.5:yellow") == [
            (0.0, "red"),
            (0.5, "yellow"),
            (0.8, "green"),
        ]

    def test_parse_score_color_map_empty(self):
        assert _parse_score_color_map("") == []

    def test_parse_score_color_map_invalid_format(self):
        with pytest.raises(ValueError):
            _parse_score_color_map("0.5red")

    def test_select_score_color_below_threshold(self):
        score_color_map = [(0.5, "red"), (0.8, "yellow"), (1.0, "green")]
        assert _select_score_color(score_color_map, 0.3) == "red"

    def test_select_score_color_at_boundary(self):
        score_color_map = [(0.5, "red"), (0.8, "yellow"), (1.0, "green")]
        assert _select_score_color(score_color_map, 0.5) == "yellow"
        assert _select_score_color(score_color_map, 0.8) == "green"

    def test_select_score_color_above_all(self):
        score_color_map = [(0.5, "red"), (0.8, "yellow"), (1.0, "green")]
        assert _select_score_color(score_color_map, 0.95) == "green"


def _make_minimal_result():
    """Return a minimal transcription result with word-level timestamps."""
    return {
        "language": "en",
        "segments": [
            {
                "start": 0.0,
                "end": 3.0,
                "text": "hello world",
                "words": [
                    {"word": "hello", "start": 0.0, "end": 0.5},
                    {"word": "world", "start": 0.6, "end": 1.0},
                ],
            }
        ],
    }


class TestSubtitlesWriterHighlight:
    """Integration tests for highlight_words + highlight_color in subtitle output."""

    def test_no_highlight_produces_plain_text(self):
        result = _make_minimal_result()
        opts = {
            "highlight_words": False,
            "max_line_width": None,
            "max_line_count": None,
            "highlight_color": None,
        }
        vtt = WriteVTT(output_dir=".")
        lines = list(vtt.iterate_result(result, opts))
        texts = [ln[2] for ln in lines]
        assert not any("<u>" in t for t in texts)
        assert not any("<font" in t for t in texts)

    def test_highlight_default_underline(self):
        result = _make_minimal_result()
        opts = {
            "highlight_words": True,
            "max_line_width": None,
            "max_line_count": None,
            "highlight_color": None,
        }
        vtt = WriteVTT(output_dir=".")
        lines = list(vtt.iterate_result(result, opts))
        texts = [ln[2] for ln in lines]
        assert any("<u>" in t for t in texts), f"Expected <u>, got: {texts}"
        assert not any("<font" in t for t in texts)

    def test_highlight_with_color(self):
        result = _make_minimal_result()
        opts = {
            "highlight_words": True,
            "max_line_width": None,
            "max_line_count": None,
            "highlight_color": "red",
        }
        vtt = WriteVTT(output_dir=".")
        lines = list(vtt.iterate_result(result, opts))
        texts = [ln[2] for ln in lines]
        assert any('<font color="red">' in t for t in texts), f"Expected font, got: {texts}"
        assert any("</font>" in t for t in texts)
        assert not any("<u>" in t for t in texts), "No <u> expected with custom color"

    def test_highlight_color_hex(self):
        result = _make_minimal_result()
        opts = {
            "highlight_words": True,
            "max_line_width": None,
            "max_line_count": None,
            "highlight_color": "#00ff00",
        }
        vtt = WriteVTT(output_dir=".")
        lines = list(vtt.iterate_result(result, opts))
        texts = [ln[2] for ln in lines]
        assert any('<font color="#00ff00">' in t for t in texts)

    def test_srt_highlight_with_color(self):
        result = _make_minimal_result()
        opts = {
            "highlight_words": True,
            "max_line_width": None,
            "max_line_count": None,
            "highlight_color": "yellow",
        }
        srt = WriteSRT(output_dir=".")
        lines = list(srt.iterate_result(result, opts))
        texts = [ln[2] for ln in lines]
        assert any('<font color="yellow">' in t for t in texts)

    def test_tag_balance(self):
        """Every opening tag must have a matching closing tag in each cue."""
        result = _make_minimal_result()
        opts = {
            "highlight_words": True,
            "max_line_width": None,
            "max_line_count": None,
            "highlight_color": "red",
        }
        srt = WriteSRT(output_dir=".")
        lines = list(srt.iterate_result(result, opts))
        for _, _, text in lines:
            assert text.count('<font color="red">') == text.count("</font>"), (
                f"Unbalanced tags in: {text!r}"
            )

    def test_tag_balance_default_underline(self):
        """Tag balance also holds for default underline mode."""
        result = _make_minimal_result()
        opts = {
            "highlight_words": True,
            "max_line_width": None,
            "max_line_count": None,
            "highlight_color": None,
        }
        srt = WriteSRT(output_dir=".")
        lines = list(srt.iterate_result(result, opts))
        for _, _, text in lines:
            assert text.count("<u>") == text.count("</u>"), (
                f"Unbalanced tags in: {text!r}"
            )

    def test_score_colors_rendered_based_on_score(self):
        """Words are colored according to their alignment score."""
        result = {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 3.0,
                    "text": "hello world test",
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.5, "score": 0.2},
                        {"word": "world", "start": 0.6, "end": 1.0, "score": 0.6},
                        {"word": "test", "start": 1.1, "end": 1.5, "score": 0.9},
                    ],
                }
            ],
        }
        opts = {
            "highlight_words": True,
            "max_line_width": None,
            "max_line_count": None,
            "highlight_color": "gray",
            "highlight_score_colors": "0.3:red,0.7:yellow,1.0:green",
        }
        vtt = WriteVTT(output_dir=".")
        lines = list(vtt.iterate_result(result, opts))
        texts = "\n".join([ln[2] for ln in lines])
        assert '<font color="red">hello</font>' in texts, f"got: {texts}"
        assert '<font color="yellow">world</font>' in texts, f"got: {texts}"
        assert '<font color="green">test</font>' in texts, f"got: {texts}"

    def test_score_colors_missing_score_falls_back_to_highlight_color(self):
        """If a word lacks a score, it should use the fixed --highlight_color."""
        result = {
            "language": "en",
            "segments": [
                {
                    "start": 0.0,
                    "end": 2.0,
                    "text": "hello world",
                    "words": [
                        {"word": "hello", "start": 0.0, "end": 0.5, "score": 0.2},
                        {"word": "world", "start": 0.6, "end": 1.0},
                    ],
                }
            ],
        }
        opts = {
            "highlight_words": True,
            "max_line_width": None,
            "max_line_count": None,
            "highlight_color": "gray",
            "highlight_score_colors": "0.3:red,0.7:yellow,1.0:green",
        }
        vtt = WriteVTT(output_dir=".")
        lines = list(vtt.iterate_result(result, opts))
        texts = "\n".join([ln[2] for ln in lines])
        assert '<font color="red">hello</font>' in texts, f"got: {texts}"
        assert '<font color="gray">world</font>' in texts, f"got: {texts}"
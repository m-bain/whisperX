"""SRT/VTT must keep segments that align() could not align (empty "words")."""

from whisperx.utils import WriteSRT, WriteVTT


def _word(text, start, end):
    return {"word": text, "start": start, "end": end, "score": 0.9}


def _result():
    # Shape returned by align() when the middle segment hits
    # "backtrack failed, resorting to original": text/start/end kept, no words.
    return {
        "language": "en",
        "segments": [
            {"start": 0.0, "end": 1.0, "text": " Hello there.",
             "words": [_word("Hello", 0.0, 0.5), _word("there.", 0.5, 1.0)]},
            {"start": 1.2, "end": 1.8, "text": " Not aligned.", "words": []},
            {"start": 2.0, "end": 3.0, "text": " Good bye.",
             "words": [_word("Good", 2.0, 2.5), _word("bye.", 2.5, 3.0)]},
        ],
    }


def _options(**overrides):
    options = {"max_line_width": None, "max_line_count": None, "highlight_words": False}
    options.update(overrides)
    return options


def test_srt_keeps_segment_without_words():
    cues = list(WriteSRT(".").iterate_result(_result(), _options()))
    assert cues == [
        ("00:00:00,000", "00:00:01,000", "Hello there."),
        ("00:00:01,200", "00:00:01,800", "Not aligned."),
        ("00:00:02,000", "00:00:03,000", "Good bye."),
    ]


def test_vtt_keeps_segment_without_words():
    cues = list(WriteVTT(".").iterate_result(_result(), _options()))
    assert [text for _, _, text in cues] == ["Hello there.", "Not aligned.", "Good bye."]


def test_segment_without_words_survives_line_wrapping():
    cues = list(WriteSRT(".").iterate_result(_result(), _options(max_line_width=12, max_line_count=1)))
    assert [text for _, _, text in cues] == ["Hello there.", "Not aligned.", "Good bye."]
    assert cues[1][:2] == ("00:00:01,200", "00:00:01,800")


def test_empty_segment_without_words_is_skipped():
    result = _result()
    result["segments"][1]["text"] = " "
    cues = list(WriteSRT(".").iterate_result(result, _options()))
    assert [text for _, _, text in cues] == ["Hello there.", "Good bye."]

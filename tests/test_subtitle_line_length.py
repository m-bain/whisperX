from whisperx.SubtitlesProcessor import SubtitlesProcessor


def _segment(text):
    words = [
        {"word": part, "start": float(index), "end": float(index) + 0.4}
        for index, part in enumerate(text.split())
    ]
    return {"start": 0.0, "end": float(len(words)), "text": text, "words": words}


def _lines(text):
    subtitles = SubtitlesProcessor([_segment(text)], "en").process_segments(True)
    return [item["text"].strip() for item in subtitles]


def test_a_line_that_fits_is_not_split():
    text = " ".join(["abcdefgh"] * 5)
    assert len(text) == 44
    assert _lines(text) == [text]


def test_a_line_past_the_limit_is_still_split():
    text = " ".join(["abcdefgh"] * 8)
    assert len(text) > 45
    lines = _lines(text)
    assert len(lines) > 1
    assert all(len(line) <= 45 for line in lines)

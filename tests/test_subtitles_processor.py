import pytest

from whisperx.SubtitlesProcessor import SubtitlesProcessor


@pytest.mark.parametrize(
    "is_vtt, separator, header",
    [(False, ",", ""), (True, ".", "WEBVTT\n\n")],
)
@pytest.mark.parametrize("advanced_splitting", [False, True])
def test_save_writes_subtitles(tmp_path, is_vtt, separator, header, advanced_splitting):
    segments = [
        {
            "start": 1.25,
            "end": 4.5,
            "text": " Hello café world. ",
            "words": [
                {"word": "Hello", "start": 1.25, "end": 2.0},
                {"word": "café", "start": 2.0, "end": 3.5},
                {"word": "world.", "start": 3.5, "end": 4.5},
            ],
        },
        {
            "start": 6.0,
            "end": 7.25,
            "text": " Goodbye. ",
            "words": [{"word": "Goodbye.", "start": 6.0, "end": 7.25}],
        },
    ]
    processor = SubtitlesProcessor(
        segments, "en", max_line_length=10, min_char_length_splitter=5, is_vtt=is_vtt
    )
    path = tmp_path / ("subtitles.vtt" if is_vtt else "subtitles.srt")

    count = processor.save(path, advanced_splitting=advanced_splitting)

    if advanced_splitting:
        expected = (
            f"1\n00:00:01{separator}250 --> 00:00:03{separator}500\nHello café\n\n"
            f"2\n00:00:03{separator}500 --> 00:00:04{separator}500\nworld.\n\n"
            f"3\n00:00:06{separator}000 --> 00:00:07{separator}250\nGoodbye.\n\n"
        )
        assert count == 3
    else:
        expected = (
            f"1\n00:00:01{separator}250 --> 00:00:04{separator}500\nHello café world.\n\n"
            f"2\n00:00:06{separator}000 --> 00:00:07{separator}250\nGoodbye.\n\n"
        )
        assert count == 2
    assert path.read_text(encoding="utf-8") == header + expected

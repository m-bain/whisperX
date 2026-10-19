"""Test that split subtitles never merge words from different speakers."""

import io

from whisperx.utils import WriteSRT, WriteVTT


def _word(word, start, end):
    return {"word": word, "start": start, "end": end}


def _diarized_result():
    return {
        "language": "en",
        "segments": [
            {
                "start": 0.0,
                "end": 1.0,
                "text": "Hello there.",
                "speaker": "SPEAKER_00",
                "words": [_word("Hello", 0.0, 0.5), _word("there.", 0.5, 1.0)],
            },
            {
                "start": 1.2,
                "end": 2.0,
                "text": "Hi back.",
                "speaker": "SPEAKER_01",
                "words": [_word("Hi", 1.2, 1.5), _word("back.", 1.5, 2.0)],
            },
            {
                "start": 2.2,
                "end": 3.0,
                "text": "How are you?",
                "speaker": "SPEAKER_01",
                "words": [
                    _word("How", 2.2, 2.4),
                    _word("are", 2.4, 2.6),
                    _word("you?", 2.6, 3.0),
                ],
            },
        ],
    }


SPLIT_OPTIONS = {"max_line_width": 42, "max_line_count": 2, "highlight_words": False}


class TestSubtitleSpeakerChange:
    def test_speaker_change_starts_new_cue(self):
        cues = list(WriteSRT(".").iterate_result(_diarized_result(), SPLIT_OPTIONS))

        assert cues == [
            ("00:00:00,000", "00:00:01,000", "[SPEAKER_00]: Hello there."),
            ("00:00:01,200", "00:00:03,000", "[SPEAKER_01]: Hi back. How are you?"),
        ]

    def test_speaker_change_starts_new_cue_with_highlight(self):
        options = dict(SPLIT_OPTIONS, highlight_words=True)
        cues = list(WriteVTT(".").iterate_result(_diarized_result(), options))

        speaker_00_cues = [text for _, _, text in cues if "SPEAKER_00" in text]
        assert speaker_00_cues
        assert all("Hi" not in text for text in speaker_00_cues)

    def test_same_speaker_segments_are_still_merged(self):
        result = _diarized_result()
        for segment in result["segments"]:
            segment["speaker"] = "SPEAKER_00"

        f = io.StringIO()
        WriteSRT(".").write_result(result, f, SPLIT_OPTIONS)

        assert f.getvalue() == (
            "1\n"
            "00:00:00,000 --> 00:00:03,000\n"
            "[SPEAKER_00]: Hello there. Hi back. How are you?\n\n"
        )

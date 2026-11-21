import os
import tempfile
from whisperx.SubtitlesProcessor import SubtitlesProcessor


def test_subtitles_processor_save_without_advanced_splitting():
    segments = [
        {
            "start": 0.0,
            "end": 2.5,
            "text": "First segment text",
            "words": [
                {"word": "First", "start": 0.0, "end": 0.8},
                {"word": "segment", "start": 0.8, "end": 1.6},
                {"word": "text", "start": 1.6, "end": 2.5},
            ],
        },
        {
            "start": 3.0,
            "end": 5.0,
            "text": "Second segment text",
            "words": [
                {"word": "Second", "start": 3.0, "end": 3.7},
                {"word": "segment", "start": 3.7, "end": 4.4},
                {"word": "text", "start": 4.4, "end": 5.0},
            ],
        },
    ]

    with tempfile.TemporaryDirectory() as tmpdir:
        # Test SRT output
        srt_path = os.path.join(tmpdir, "output.srt")
        processor_srt = SubtitlesProcessor(segments, lang="en", is_vtt=False)
        count_srt = processor_srt.save(srt_path, advanced_splitting=False)

        assert count_srt == 2
        with open(srt_path, "r", encoding="utf-8") as f:
            content_srt = f.read()

        assert "1\n00:00:00,000 --> 00:00:02,500\nFirst segment text\n\n" in content_srt
        assert "2\n00:00:03,000 --> 00:00:05,000\nSecond segment text\n\n" in content_srt

        # Test VTT output
        vtt_path = os.path.join(tmpdir, "output.vtt")
        processor_vtt = SubtitlesProcessor(segments, lang="en", is_vtt=True)
        count_vtt = processor_vtt.save(vtt_path, advanced_splitting=False)

        assert count_vtt == 2
        with open(vtt_path, "r", encoding="utf-8") as f:
            content_vtt = f.read()

        assert content_vtt.startswith("WEBVTT\n\n")
        assert "1\n00:00:00.000 --> 00:00:02.500\nFirst segment text\n\n" in content_vtt
        assert "2\n00:00:03.000 --> 00:00:05.000\nSecond segment text\n\n" in content_vtt

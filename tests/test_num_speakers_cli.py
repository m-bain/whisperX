"""Test that the CLI exposes --num_speakers and guards it against min/max."""

import subprocess
import sys


def _run_cli(*args):
    """Invoke the CLI in a subprocess and return the completed process.

    Argument validation happens before any model is loaded, so these calls
    return quickly and need no network access or model downloads.
    """
    return subprocess.run(
        [sys.executable, "-m", "whisperx", *args],
        capture_output=True,
        text=True,
    )


def test_num_speakers_is_exposed_in_help():
    result = _run_cli("--help")
    assert result.returncode == 0
    assert "--num_speakers" in result.stdout


def test_num_speakers_rejects_min_speakers():
    result = _run_cli("audio.wav", "--diarize", "--num_speakers", "2", "--min_speakers", "2")
    assert result.returncode == 2
    assert "--num_speakers cannot be combined" in result.stderr


def test_num_speakers_rejects_max_speakers():
    result = _run_cli("audio.wav", "--diarize", "--num_speakers", "2", "--max_speakers", "3")
    assert result.returncode == 2
    assert "--num_speakers cannot be combined" in result.stderr


def test_min_and_max_speakers_still_allowed_together():
    """The existing range form must keep working unchanged."""
    result = _run_cli("--help")
    assert "--min_speakers" in result.stdout
    assert "--max_speakers" in result.stdout

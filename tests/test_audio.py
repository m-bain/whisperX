import wave

import numpy as np

from whisperx.audio import SAMPLE_RATE, load_audio


def _write_wav(path, sr=SAMPLE_RATE, seconds=1):
    # A short mono 16-bit PCM tone, enough to exercise both decode paths.
    samples = (np.sin(np.linspace(0, 2 * np.pi * 220 * seconds, sr * seconds)) * 10000).astype(np.int16)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sr)
        w.writeframes(samples.tobytes())


def test_load_audio_use_tmp_file_matches_in_memory(tmp_path):
    wav_path = tmp_path / "fixture.wav"
    _write_wav(wav_path)

    in_memory = load_audio(str(wav_path))
    via_tmp_file = load_audio(str(wav_path), use_tmp_file=True)

    # The temp-file path must produce exactly the same waveform as the default in-memory path.
    assert via_tmp_file.dtype == np.float32
    assert np.array_equal(in_memory, via_tmp_file)

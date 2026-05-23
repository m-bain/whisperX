from pathlib import Path
import unittest
from unittest.mock import patch, MagicMock

import numpy as np
import torch

from whisperx.audio import load_audio, log_mel_spectrogram
from whisperx.utils import WriteTXT


class TestPathlibSupport(unittest.TestCase):

    @patch('subprocess.run')
    def test_load_audio_pathlib(self, mock_run):
        mock_run.return_value.stdout = np.zeros(16000, dtype=np.int16).tobytes()
        
        # This should accept a Path object and convert it to string path
        audio_path = Path("test_audio.wav")
        load_audio(audio_path)
        
        # Verify that subprocess.run was called with string path
        called_args = mock_run.call_args[0][0]
        self.assertIn("test_audio.wav", called_args)

    @patch('whisperx.audio.load_audio')
    def test_log_mel_spectrogram_pathlib(self, mock_load_audio):
        mock_load_audio.return_value = np.zeros(16000, dtype=np.float32)
        
        # This should accept a Path object and invoke load_audio
        audio_path = Path("test_audio.wav")
        log_mel_spectrogram(audio_path, n_mels=80)
        
        mock_load_audio.assert_called_once_with(audio_path)

    def test_result_writer_pathlib(self):
        writer = WriteTXT(Path("/tmp/output"))
        self.assertEqual(writer.output_dir, str(Path("/tmp/output")))

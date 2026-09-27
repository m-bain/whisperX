import os
from pathlib import Path
from unittest.mock import patch, MagicMock
import numpy as np
import pytest

from whisperx.audio import load_audio

def test_load_audio_with_path_object():
    # Arrange
    test_path = Path("fake/audio/path.wav")
    
    # 1000 samples of 16-bit PCM integer zeros
    dummy_bytes = b"\x00\x00" * 1000
    
    mock_completed_process = MagicMock()
    mock_completed_process.stdout = dummy_bytes
    
    with patch("subprocess.run") as mock_run:
        mock_run.return_value = mock_completed_process
        
        # Act
        result = load_audio(test_path, sr=16000)
        
        # Assert
        # 1. Assert no TypeError was raised and result is np.ndarray
        assert isinstance(result, np.ndarray)
        assert result.dtype == np.float32
        assert len(result) == 1000
        
        # 2. Assert subprocess.run was called
        mock_run.assert_called_once()
        
        # 3. Assert that the path was converted to a string in cmd list
        called_args = mock_run.call_args[0][0]
        # Find index after "-i" flag
        idx = called_args.index("-i")
        passed_file_path = called_args[idx + 1]
        
        assert isinstance(passed_file_path, str)
        assert passed_file_path == os.fspath(test_path)

"""
Audio processing utilities for WhisperX.

This module contains functions for loading, processing, and transforming audio data
for use with the WhisperX speech recognition system.
"""

import os
import subprocess
from functools import lru_cache
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from whisperx.utils import exact_div

# Audio hyperparameters
SAMPLE_RATE = 16000  # 16kHz audio sample rate
N_FFT = 400  # Size of FFT window
HOP_LENGTH = 160  # Hop length for STFT (10ms at 16kHz)
CHUNK_LENGTH = 30  # Length of audio chunks in seconds
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # Each token represents 20ms of audio (initial convolutions have stride 2)
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token


def load_audio(file: str, sr: int = SAMPLE_RATE) -> np.ndarray:
    """
    Open an audio file and read as mono waveform, resampling as necessary.

    This function uses ffmpeg to decode audio, converting to mono and resampling
    to the target sample rate.

    Parameters
    ----------
    file : str
        The path to the audio file to open
    sr : int, default=16000
        The target sample rate to resample the audio to if necessary

    Returns
    -------
    np.ndarray
        A NumPy array containing the audio waveform, in float32 dtype normalized to [-1.0, 1.0]

    Raises
    ------
    RuntimeError
        If ffmpeg fails to load the audio file
    """
    try:
        # Use ffmpeg to decode audio while down-mixing and resampling as necessary
        cmd = [
            "ffmpeg",
            "-nostdin",  # No interactive stdin
            "-threads", "0",  # Use optimal number of threads
            "-i", file,  # Input file
            "-f", "s16le",  # Output format: signed 16-bit little-endian
            "-ac", "1",  # Convert to mono
            "-acodec", "pcm_s16le",  # PCM signed 16-bit little-endian codec
            "-ar", str(sr),  # Target sample rate
            "-",  # Output to stdout
        ]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    # Convert to float32 and normalize to [-1.0, 1.0]
    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim an array to a specific length along a specified axis.

    This function is used to ensure audio inputs have a consistent length
    as expected by the encoder.

    Parameters
    ----------
    array : np.ndarray or torch.Tensor
        The input array to pad or trim
    length : int, default=N_SAMPLES
        The target length for the array
    axis : int, default=-1
        The axis along which to pad or trim

    Returns
    -------
    np.ndarray or torch.Tensor
        The padded or trimmed array with the same type as the input
    """
    if torch.is_tensor(array):
        # Trim if necessary
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        # Pad if necessary
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        # Trim if necessary
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        # Pad if necessary
        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    """
    Load the mel filterbank matrix for projecting STFT into a Mel spectrogram.

    The filterbank matrices are pre-computed and stored in the assets directory
    to remove the librosa dependency.

    Parameters
    ----------
    device : torch.device
        The device to load the filters to
    n_mels : int
        The number of Mel-frequency filters (80 or 128)

    Returns
    -------
    torch.Tensor
        The mel filterbank matrix on the specified device

    Raises
    ------
    AssertionError
        If n_mels is not 80 or 128
    """
    assert n_mels in [80, 128], f"Unsupported n_mels: {n_mels}, must be 80 or 128"
    with np.load(
        os.path.join(os.path.dirname(__file__), "assets", "mel_filters.npz")
    ) as f:
        return torch.from_numpy(f[f"mel_{n_mels}"]).to(device)


def log_mel_spectrogram(
    audio: Union[str, np.ndarray, torch.Tensor],
    n_mels: int,
    padding: int = 0,
    device: Optional[Union[str, torch.device]] = None,
):
    """
    Compute the log-Mel spectrogram of audio input.

    This function converts audio waveform to a log-scaled Mel spectrogram
    which is the input format expected by Whisper models.

    Parameters
    ----------
    audio : Union[str, np.ndarray, torch.Tensor]
        The audio input, which can be:
        - A path to an audio file
        - A NumPy array containing the audio waveform
        - A PyTorch tensor containing the audio waveform
        The audio should be sampled at 16 kHz
    n_mels : int
        The number of Mel-frequency filters (80 or 128)
    padding : int, default=0
        Number of zero samples to pad to the right
    device : Optional[Union[str, torch.device]], default=None
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (n_mels, n_frames)
        A tensor containing the normalized log Mel spectrogram
    """
    # Convert to torch tensor if necessary
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    # Move to device and apply padding if needed
    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    
    # Compute STFT
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    # Apply mel filterbank
    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    # Convert to log scale and normalize
    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    
    return log_spec

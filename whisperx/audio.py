import os
import subprocess
from functools import lru_cache
from typing import Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torchaudio.functional import resample

from .utils import exact_div

# hard-coded audio hyperparameters
SAMPLE_RATE = 16000
ALPHA = 0.5
N_FFT = 400
HOP_LENGTH = 160
CHUNK_LENGTH = 30
N_SAMPLES = CHUNK_LENGTH * SAMPLE_RATE  # 480000 samples in a 30-second chunk
N_FRAMES = exact_div(N_SAMPLES, HOP_LENGTH)  # 3000 frames in a mel spectrogram input

N_SAMPLES_PER_TOKEN = HOP_LENGTH * 2  # the initial convolutions has stride 2
FRAMES_PER_SECOND = exact_div(SAMPLE_RATE, HOP_LENGTH)  # 10ms per audio frame
TOKENS_PER_SECOND = exact_div(SAMPLE_RATE, N_SAMPLES_PER_TOKEN)  # 20ms per audio token


def load_audio(file: str, sr: int = SAMPLE_RATE):
    """
    Open an audio file and read as mono waveform, resampling as necessary

    Parameters
    ----------
    file: str
        The audio file to open

    sr: int
        The sample rate to resample the audio if necessary

    Returns
    -------
    A NumPy array containing the audio waveform, in float32 dtype.
    """
    try:
        # Launches a subprocess to decode audio while down-mixing and resampling as necessary.
        # Requires the ffmpeg CLI to be installed.
        cmd = [
            "ffmpeg",
            "-nostdin",
            "-threads",
            "0",
            "-i",
            file,
            "-f",
            "s16le",
            "-ac",
            "1",
            "-acodec",
            "pcm_s16le",
            "-ar",
            str(sr),
            "-",
        ]
        out = subprocess.run(cmd, capture_output=True, check=True).stdout
    except subprocess.CalledProcessError as e:
        raise RuntimeError(f"Failed to load audio: {e.stderr.decode()}") from e

    return np.frombuffer(out, np.int16).flatten().astype(np.float32) / 32768.0


def resample_audio(audio: Union[torch.Tensor, np.ndarray ], sample_rate: int) -> torch.Tensor:
    """
    Resample audio: np.ndarray to 16 kHz

    Parameters
    ----------
    audio: Union[np.ndarray, torch.Tensor]
        The data to be resampled, 1D (mono) or 2D (stereo). This parameter can accept either a NumPy array (np.ndarray) or a PyTorch tensor (torch.Tensor) containing audio data. The audio data should be of type float32, float64, int16, or int32.
    audio: np.ndarray[ float32 | float64 | int16 | int32 ]
        The data to be resampled, 1D(mono) or 2D(stereo)

    sample_rate: int 
        The sample rate of audio

    Returns
    -------
    A torch Tensor 1D containing the audio waveform, in float32 dtype.
    """
    if type(audio) != torch.Tensor:
        audio = torch.from_numpy(audio)

    if audio.dtype not in (torch.float32, torch.float64, torch.int16, torch.int32):
        raise ValueError(f"Audio type must be one of [float32, float64, int16, int32], not {audio.dtype}")

    audio_dtype = audio.dtype

    if audio.ndim == 2: #Stereo
        if audio.shape[0] == 2 or audio.shape[1] == 2:
            if audio.shape[1] == 2: #SciPy | Soundfile
                audio = torch.transpose(audio, 0, 1)

            # Convert to mono
            # MIX = A * (1 - ALPHA) + B * ALPHA
            audio = (audio[0] * ALPHA + audio[1] * ALPHA)

        elif audio.shape[0] != 1:
            raise ValueError(f"Invalid audio shape ({audio.shape}). Audio must be provided as: \n"
                             "([channel, time], [time]) for mono audio, \n"
                             "([channel, time], [time, channel]) for stereo audio")

    elif audio.ndim != 1:
        raise ValueError(f"Audio ndim must be 1D(mono) or 2D(stereo)")


    if audio_dtype in (torch.int16, torch.int32):
        audio = audio.to(torch.float32) / (32768.0 if audio_dtype == torch.int16 else 2147483648.0)
    elif audio_dtype == torch.float64:
        audio = audio.to(torch.float32)

    return resample(audio, sample_rate, SAMPLE_RATE).flatten()


def pad_or_trim(array, length: int = N_SAMPLES, *, axis: int = -1):
    """
    Pad or trim the audio array to N_SAMPLES, as expected by the encoder.
    """
    if torch.is_tensor(array):
        if array.shape[axis] > length:
            array = array.index_select(
                dim=axis, index=torch.arange(length, device=array.device)
            )

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = F.pad(array, [pad for sizes in pad_widths[::-1] for pad in sizes])
    else:
        if array.shape[axis] > length:
            array = array.take(indices=range(length), axis=axis)

        if array.shape[axis] < length:
            pad_widths = [(0, 0)] * array.ndim
            pad_widths[axis] = (0, length - array.shape[axis])
            array = np.pad(array, pad_widths)

    return array


@lru_cache(maxsize=None)
def mel_filters(device, n_mels: int) -> torch.Tensor:
    """
    load the mel filterbank matrix for projecting STFT into a Mel spectrogram.
    Allows decoupling librosa dependency; saved using:

        np.savez_compressed(
            "mel_filters.npz",
            mel_80=librosa.filters.mel(sr=16000, n_fft=400, n_mels=80),
        )
    """
    assert n_mels in [80, 128], f"Unsupported n_mels: {n_mels}"
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
    Compute the log-Mel spectrogram of

    Parameters
    ----------
    audio: Union[str, np.ndarray, torch.Tensor], shape = (*)
        The path to audio or either a NumPy array or Tensor containing the audio waveform in 16 kHz

    n_mels: int
        The number of Mel-frequency filters, only 80 is supported

    padding: int
        Number of zero samples to pad to the right

    device: Optional[Union[str, torch.device]]
        If given, the audio tensor is moved to this device before STFT

    Returns
    -------
    torch.Tensor, shape = (80, n_frames)
        A Tensor that contains the Mel spectrogram
    """
    if not torch.is_tensor(audio):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio = torch.from_numpy(audio)

    if device is not None:
        audio = audio.to(device)
    if padding > 0:
        audio = F.pad(audio, (0, padding))
    window = torch.hann_window(N_FFT).to(audio.device)
    stft = torch.stft(audio, N_FFT, HOP_LENGTH, window=window, return_complex=True)
    magnitudes = stft[..., :-1].abs() ** 2

    filters = mel_filters(audio.device, n_mels)
    mel_spec = filters @ magnitudes

    log_spec = torch.clamp(mel_spec, min=1e-10).log10()
    log_spec = torch.maximum(log_spec, log_spec.max() - 8.0)
    log_spec = (log_spec + 4.0) / 4.0
    return log_spec

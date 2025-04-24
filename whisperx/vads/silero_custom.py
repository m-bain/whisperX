from io import IOBase
from pathlib import Path
from typing import Mapping, Text, Optional, Union, List, Tuple

import torch
import torchaudio

from whisperx.diarize import Segment as SegmentX
from whisperx.vads.vad import Vad

AudioFile = Union[Text, Path, IOBase, Mapping]


class SileroCustom(Vad):
    """
    Voice Activity Detection using Silero VAD model.
    
    This class implements a custom wrapper around the Silero VAD model to detect
    speech segments in audio files. It processes audio to identify where speech occurs
    and returns time-stamped segments.
    """
    
    def __init__(self, **kwargs):
        """
        Initialize the Silero VAD model.
        
        Args:
            kwargs: Keyword arguments including:
                - vad_onset: Threshold for voice detection
                - chunk_size: Size of audio chunks for processing
                - vad_onnx: Whether to use ONNX version of the model
                - silero_merge_cutoff: Maximum gap between segments to merge them
        """
        print(">>Performing voice activity detection using Silero...")
        super().__init__(kwargs['vad_onset'])

        self.vad_onset = kwargs['vad_onset']
        self.chunk_size = kwargs['chunk_size']
        self.vad_onnx = kwargs['vad_onnx']
        self.merge_cutoff = kwargs['silero_merge_cutoff']
        
        self.vad_pipeline, vad_utils = torch.hub.load(
            repo_or_dir='snakers4/silero-vad',
            model='silero_vad',
            force_reload=False,
            onnx=self.vad_onnx,
            trust_repo=True
        )
        (self.get_speech_timestamps, _, self.read_audio, _, _) = vad_utils

    def __call__(self, audio: AudioFile, **kwargs) -> List[SegmentX]:
        """
        Process audio to detect speech segments.
        
        Args:
            audio: Audio data containing waveform and sample rate
            kwargs: Additional arguments
            
        Returns:
            List of speech segments with start and end times
        """
        # Only accept 16000 Hz for now.
        # Note: Silero models support both 8000 and 16000 Hz. Although other values are not directly supported,
        # multiples of 16000 (e.g. 32000 or 48000) are cast to 16000 inside of the JIT model!
        sample_rate = audio["sample_rate"]
        waveform = torch.from_numpy(audio["waveform"])
        
        # Ensure audio is at 16kHz for Silero VAD
        if sample_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sample_rate, new_freq=16000)
            waveform = resampler(waveform)
            sample_rate = 16000
        
        # Process audio with Silero VAD
        predicts = self.vad_pipeline.audio_forward(waveform, sr=16000)[0]
        
        # Handle mono audio (single channel)
        voice_activity = predicts > self.vad_onset
        segments = self._get_segments_from_activity(voice_activity)
        
        return [SegmentX(start, end, "UNKNOWN") for start, end in segments]
    
    def _get_segments_from_activity(self, voice_activity) -> List[Tuple[float, float]]:
        """
        Convert voice activity tensor to time segments.
        
        Args:
            voice_activity: Tensor of boolean values indicating voice activity
            
        Returns:
            List of tuples containing (start_time, end_time) in seconds
        """
        segments = []
        start_idx = None
        
        # Find continuous voice segments
        for i, is_voice in enumerate(voice_activity):
            if is_voice and start_idx is None:
                start_idx = i
            elif not is_voice and start_idx is not None:
                # Convert chunk indices to seconds (each chunk is 32ms)
                start_time = round(start_idx * 32 / 1000, 3)
                end_time = round(i * 32 / 1000, 3)
                
                # Store segment without applying max duration constraint
                segments.append((start_time, end_time))
                start_idx = None
        
        # Handle case where voice activity continues until end
        if start_idx is not None:
            start_time = round(start_idx * 32 / 1000, 3)
            end_time = round(len(voice_activity) * 32 / 1000, 3)
            segments.append((start_time, end_time))
        
        # Merge segments that are close to each other (less than merge_cutoff seconds apart)
        if len(segments) > 1:
            merged_segments = []
            current_segment = segments[0]
            
            for i in range(1, len(segments)):
                # Calculate the gap between current segment end and next segment start
                gap = segments[i][0] - current_segment[1]
                
                if gap < self.merge_cutoff:
                    # Merge the segments by extending the end time
                    current_segment = (current_segment[0], segments[i][1])
                else:
                    # Gap is too large, add current segment and start a new one
                    merged_segments.append(current_segment)
                    current_segment = segments[i]
            
            # Add the last segment
            merged_segments.append(current_segment)
            segments = merged_segments
        
        return segments

    @staticmethod
    def preprocess_audio(audio):
        """
        Preprocess audio for VAD processing.
        
        Args:
            audio: Audio data
            
        Returns:
            Preprocessed audio data
        """
        return audio

    @staticmethod
    def merge_chunks(
        segments_list,
        chunk_size,
        onset: float = 0.5,
        offset: Optional[float] = None,
        individual_segment: bool = True
    ):
        """
        Merge speech segments into chunks for processing.
        
        Args:
            segments_list: List of speech segments
            chunk_size: Maximum size of each chunk in seconds
            onset: Voice activity detection onset threshold
            offset: Voice activity detection offset threshold
            individual_segment: Whether to process each segment individually
            
        Returns:
            List of merged segments
        """
        assert chunk_size > 0
        if len(segments_list) == 0:
            print("No active speech found in audio")
            return []
        assert segments_list, "segments_list is empty."
        return Vad.merge_chunks(segments_list, chunk_size, onset, offset, individual_segment=True)

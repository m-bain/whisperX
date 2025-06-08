import numpy as np
import pandas as pd
from pyannote.audio import Pipeline
from typing import Optional, Union
import torch

from whisperx.audio import load_audio, SAMPLE_RATE
from whisperx.types import TranscriptionResult, AlignedTranscriptionResult


class DiarizationPipeline:
    def __init__(
        self,
        model_name=None,
        use_auth_token=None,
        device: Optional[Union[str, torch.device]] = "cpu",
    ):
        if isinstance(device, str):
            device = torch.device(device)
        model_config = model_name or "pyannote/speaker-diarization-3.1"
        self.model = Pipeline.from_pretrained(model_config, use_auth_token=use_auth_token).to(device)

    def __call__(
        self,
        audio: Union[str, np.ndarray],
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE
        }
        segments = self.model(audio_data, num_speakers = num_speakers, min_speakers=min_speakers, max_speakers=max_speakers)
        diarize_df = pd.DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
        return diarize_df


def assign_word_speakers(
    diarize_df: pd.DataFrame,
    transcript_result: Union[AlignedTranscriptionResult, TranscriptionResult],
    fill_nearest=False,
) -> dict:
    transcript_segments = transcript_result["segments"]
    for seg in transcript_segments:
        # assign speaker to segment (if any)
        diarize_df['intersection'] = np.minimum(diarize_df['end'], seg['end']) - np.maximum(diarize_df['start'], seg['start'])
        diarize_df['union'] = np.maximum(diarize_df['end'], seg['end']) - np.minimum(diarize_df['start'], seg['start'])
        
        intersected = diarize_df[diarize_df["intersection"] > 0]

        speaker = None
        if len(intersected) > 0:
            # Choosing most strong intersection
            speaker = intersected.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
        elif fill_nearest:
            # Otherwise choosing closest
            speaker = diarize_df.sort_values(by=["intersection"], ascending=False)["speaker"].values[0]

        if speaker is not None:
            seg["speaker"] = speaker
        
        # assign speaker to words
        if 'words' in seg:
            for word in seg['words']:
                if 'start' in word:
                    diarize_df['intersection'] = np.minimum(diarize_df['end'], word['end']) - np.maximum(diarize_df['start'], word['start'])
                    diarize_df['union'] = np.maximum(diarize_df['end'], word['end']) - np.minimum(diarize_df['start'], word['start'])
                    
                    intersected = diarize_df[diarize_df["intersection"] > 0]

                    word_speaker = None
                    if len(intersected) > 0:
                        # Choosing most strong intersection
                        word_speaker = intersected.groupby("speaker")["intersection"].sum().sort_values(ascending=False).index[0]
                    elif fill_nearest:
                        # Otherwise choosing closest
                        word_speaker = diarize_df.sort_values(by=["intersection"], ascending=False)["speaker"].values[0]

                    if word_speaker is not None:
                        word["speaker"] = word_speaker
        
    return transcript_result            


class Segment:
    def __init__(self, start:int, end:int, speaker:Optional[str]=None):
        self.start = start
        self.end = end
        self.speaker = speaker

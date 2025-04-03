import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from pyannote.audio import Pipeline

from .audio import load_audio, SAMPLE_RATE

class OfflineDiarizationPipeline:
    def __init__(
        self,
        config_path,
        device="cpu",
    ):
        if isinstance(device, str):
            device = torch.device(device)
            
        # Load the pipeline with local config
        self.model = self._load_pipeline_from_pretrained(config_path).to(device)

    def _load_pipeline_from_pretrained(self, path_to_config):
        path_to_config = Path(path_to_config)
        
        if not path_to_config.exists():
            raise FileNotFoundError(f"Config file not found: {path_to_config}")
        
        print(f"Loading pyannote pipeline from {path_to_config}...")
        # the paths in the config are relative to the current working directory
        # so we need to change the working directory to the model path
        # and then change it back
        
        cwd = Path.cwd().resolve()  # store current working directory
        
        # first .parent is the folder of the config, second .parent is the folder containing the 'models' folder
        cd_to = path_to_config.parent.parent.resolve()
        
        print(f"Changing working directory to {cd_to}")
        os.chdir(cd_to)
        
        pipeline = Pipeline.from_pretrained(path_to_config)
        
        print(f"Changing working directory back to {cwd}")
        os.chdir(cwd)
        
        return pipeline

    def __call__(
        self,
        audio,
        num_speakers=None,
        min_speakers=None,
        max_speakers=None,
    ):
        if isinstance(audio, str):
            audio = load_audio(audio)
        audio_data = {
            'waveform': torch.from_numpy(audio[None, :]),
            'sample_rate': SAMPLE_RATE
        }
        segments = self.model(audio_data, num_speakers=num_speakers, min_speakers=min_speakers, max_speakers=max_speakers)
        diarize_df = pd.DataFrame(segments.itertracks(yield_label=True), columns=['segment', 'label', 'speaker'])
        diarize_df['start'] = diarize_df['segment'].apply(lambda x: x.start)
        diarize_df['end'] = diarize_df['segment'].apply(lambda x: x.end)
        return diarize_df

import os
from typing import List
from whisperx.prosody_features.utils import generate_char_frame_sequence
import gc
import json
import tqdm
import argparse
from whisperx.transcribe import load_model
from whisperx.alignment import load_align_model, align_for_prosody_features
from whisperx.audio import load_audio
import torch
import torch.distributed as dist
from torch.multiprocessing import Process

MODEL_DIR = "/project/shrikann_35/nmehlman/vpc/models" # TODO make this non hard coded

def init_process(rank, size, fn, *args):
    """ Initialize the distributed environment. """
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("gloo", rank=rank, world_size=size)
    fn(rank, size, *args)

def get_aligned_chars(
    rank,
    size,
    whisper_model,
    alignment_model,
    alignmet_model_metadata,
    audio_file: str,
    device: str = "cpu",
) -> List[dict]:

    batch_size = 4  # reduce if low on GPU mem

    audio = load_audio(audio_file)
    trans_result = whisper_model.transcribe(audio, batch_size=batch_size, language="en")
    
    try: 
        align_result = align_for_prosody_features(
            trans_result["segments"],
            alignment_model,
            alignmet_model_metadata,
            audio,
            device,
        )
        # Save or process align_result as needed
    except Exception as e:
        print(f"Error in alignment: {e}")

def run(rank, size):
    # Load models
    whisper_model = load_model(MODEL_DIR)
    alignment_model, alignmet_model_metadata = load_align_model(MODEL_DIR)

    # Set device for each process
    device = f"cuda:{rank}" if torch.cuda.is_available() else "cpu"

    # Call the function
    get_aligned_chars(rank, size, whisper_model, alignment_model, alignmet_model_metadata, "path_to_audio_file", device)

if __name__ == "__main__":
    size = torch.cuda.device_count()
    processes = []
    for rank in range(size):
        p = Process(target=init_process, args=(rank, size, run))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
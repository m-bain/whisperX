import os
from typing import List
import whisperx
from whisperx.prosody_features.utils import generate_char_frame_sequence
import gc
import json
import tqdm
import argparse
from whisperx.transcribe import load_model
from whisperx.alignment import load_align_model, align_for_prosody_features
from whisperx.audio import load_audio

MODEL_DIR = "/project/shrikann_35/nmehlman/vpc/models"


def get_aligned_chars(
    whisper_model,
    alignment_model,
    alignmet_model_metadata,
    audio_file: str,
    device: str = "cpu",
) -> List[dict]:

    batch_size = 16  # reduce if low on GPU mem

    audio = load_audio(audio_file)
    trans_result = whisper_model.transcribe(audio, batch_size=batch_size, language="en")
    
    align_result = align_for_prosody_features(
        trans_result["segments"],
        alignment_model,
        alignmet_model_metadata,
        audio,
        device,
        return_char_alignments=True,
    )

    chars = align_result["char_segments"]
    
    return chars
   


if __name__ == "__main__":

    # Argument parser setup
    parser = argparse.ArgumentParser(
        description="Feature extraction script with alignment."
    )
    parser.add_argument(
        "--data-root",
        type=str,
        help="Root data directory.",
    )
    parser.add_argument(
        "--save-root",
        type=str,
        help="Root directory for saving prosody features.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for model inference. Default: 'cuda'.",
    )
    parser.add_argument(
        "--compute_type",
        type=str,
        default="float16",
        help="Type of compute format to use. Default: 'float16'.",
    )

    args = parser.parse_args()
    device = args.device
    compute_type = args.compute_type
    data_root = args.data_root
    save_root = args.save_root

    # Create mirror directory structure
    for dirpath, dirnames, filenames in os.walk(data_root):
        print(dirpath)
        structure = os.path.join(save_root, os.path.relpath(dirpath, data_root))
        if not os.path.isdir(structure):
            os.makedirs(structure)

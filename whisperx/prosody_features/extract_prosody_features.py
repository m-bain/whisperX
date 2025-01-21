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

MODEL_DIR = "/project/shrikann_35/nmehlman/vpc/models"


def get_aligned_chars(
    whisper_model,
    alignment_model,
    alignmet_model_metadata,
    audio_file: str,
    device: str = "cpu",
) -> List[dict]:

    batch_size = 4  # reduce if low on GPU mem

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
    parser.add_argument(
        "--file-type",
        type=str,
        default="wav",
        help="Type audio file. Default: 'wav'.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip processing of existing files.",
    )

    # Parse args
    args = parser.parse_args()
    device = args.device
    compute_type = args.compute_type
    data_root = args.data_root
    save_root = args.save_root
    file_type = args.file_type
    skip_existing = args.skip_existing

    # Pre-load models
    whisper_model = load_model("large-v2", device, compute_type=compute_type)
    alignment_model, alignmet_model_metadata = load_align_model(
        language_code="en", device=device
    )

    bad_files = []

    # Create mirror directory structure
    for dirpath, dirnames, filenames in os.walk(data_root):
        rel_path = os.path.relpath(dirpath, data_root)
        save_dir_path = os.path.join(save_root, rel_path)
        if not os.path.isdir(save_dir_path):
            os.makedirs(save_dir_path)

        if filenames: # If the directory contains audio files
            
            audio_files = [f for f in filenames if f.endswith(file_type)]
            for file in tqdm.tqdm(audio_files, desc=f'extracting features for {rel_path}'): # For each audio file in the directory                    
                
                audio_file_path = os.path.join(dirpath, file)
                save_path = os.path.join(save_dir_path, file.replace(file_type, ".json"))

                # Skip previously generated files
                if os.path.exists(save_path) and skip_existing: 
                    continue

                # Perform alignment and generate char sequence feature
                aligned_chars = get_aligned_chars(
                    whisper_model=whisper_model,
                    alignment_model=alignment_model,
                    alignmet_model_metadata=alignmet_model_metadata,
                    audio_file=audio_file_path,
                    device=device,
                )

                # Handels error cases
                if aligned_chars == []:
                    print("ERROR: failed to align file")
                    bad_files.append(audio_file_path)
                    with open(os.path.join(save_root, 'bad_files.json'), "w") as save_file:
                        json.dump(bad_files, save_file)
                    continue

                char_seq = generate_char_frame_sequence(aligned_chars)

                if char_seq is None:
                    print("ERROR: failed to generate char sequence")
                    bad_files.append(audio_file_path)
                    with open(os.path.join(save_root, 'bad_files.json'), "w") as save_file:
                        json.dump(bad_files, save_file)
                    continue

                # Save
                with open(save_path, "w") as save_file:
                    json.dump(char_seq, save_file)


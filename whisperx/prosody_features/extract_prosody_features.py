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
    parser.add_argument(
        "--chunk-file",
        type=str,
        required=False,
        help="JSON file containing all the chunks.",
    )
    parser.add_argument(
        "--chunk-index",
        type=int,
        required=False,
        help="Index of the chunk to process.",
    )

    # Parse args
    args = parser.parse_args()
    device = args.device
    compute_type = args.compute_type
    data_root = args.data_root
    save_root = args.save_root
    file_type = args.file_type
    skip_existing = args.skip_existing
    chunk_file = args.chunk_file
    chunk_idx = args.chunk_index

    # Pre-load models
    whisper_model = load_model("large-v2", device, compute_type=compute_type)
    alignment_model, alignmet_model_metadata = load_align_model(
        language_code="en", device=device
    )

    if chunk_file:
        assert chunk_idx is not None, "chunk-index must be provided if chunk_file is specified."
        print('Extracting chunk index:', chunk_idx)
        # Load the full chunk file
        with open(args.chunk_file, "r") as f:
            all_chunks = json.load(f)

        # Get the specific chunk to process
        chunk_index = args.chunk_index
        chunk_files = all_chunks[chunk_index]

        # Prepare the save paths and ensure directory structure mirrors the input
        all_audio_files = []
        
        for audio_file_path in chunk_files:
            # Determine relative path for mirroring directory structure
            rel_path = os.path.relpath(audio_file_path, data_root)
            save_path = os.path.join(save_root, rel_path.replace(file_type, ".json"))

            # Ensure the directory exists
            os.makedirs(os.path.dirname(save_path), exist_ok=True)

            # Add file paths for processing
            all_audio_files.append((audio_file_path, save_path))
            
    else: # Locate all audio files
        all_audio_files = []
        for dirpath, dirnames, filenames in os.walk(data_root):
            rel_path = os.path.relpath(dirpath, data_root)
            save_dir_path = os.path.join(save_root, rel_path)
            if not os.path.isdir(save_dir_path):
                os.makedirs(save_dir_path)

            audio_files = [f for f in filenames if f.endswith(file_type)]
            for file in audio_files:
                audio_file_path = os.path.join(dirpath, file)
                save_path = os.path.join(save_dir_path, file.replace(file_type, ".json"))
                all_audio_files.append((audio_file_path, save_path))

    # Process all files
    bad_files = []
    
    if chunk_file:
        bad_file_log = os.path.join(save_root, f'bad_files_{chunk_idx}.json')
    else:
        bad_file_log = os.path.join(save_root, 'bad_files.json')
    
    for audio_file_path, save_path in tqdm.tqdm(all_audio_files, desc='Extracting features'):

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

        # Handle error cases
        if aligned_chars == []:
            print("ERROR: failed to align file")
            bad_files.append(audio_file_path)
            with open(bad_file_log, "w") as save_file:
                json.dump(bad_files, save_file)
            continue

        char_seq = generate_char_frame_sequence(aligned_chars)

        if char_seq is None:
            print("ERROR: failed to generate char sequence")
            bad_files.append(audio_file_path)
            with open(bad_file_log, "w") as save_file:
                json.dump(bad_files, save_file)
            continue

        # Save
        with open(save_path, "w") as save_file:
            print(f"Saving prosody features to {save_path}")
            json.dump(char_seq, save_file)

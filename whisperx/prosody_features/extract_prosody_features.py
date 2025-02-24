import os
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
from typing import List
from whisperx.prosody_features.utils import generate_char_frame_sequence
import json
import tqdm
import argparse
from whisperx.transcribe import load_model
from whisperx.alignment import load_align_model, align_for_prosody_features
from whisperx.audio import load_audio

MODEL_DIR = "/project/shrikann_35/nmehlman/vpc/models"


def setup(rank, world_size):
    """Initialize the distributed process group."""
    os.environ["MASTER_ADDR"] = "127.0.0.1"  # Localhost for single-node
    os.environ["MASTER_PORT"] = "12355"  # Choose any open port
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)


def cleanup():
    """Destroy the distributed process group."""
    dist.destroy_process_group()


def get_aligned_chars(
    whisper_model,
    alignment_model,
    alignmet_model_metadata,
    audio_file: str,
    device: str = "cpu",
) -> List[dict]:
    """Perform transcription and alignment for a given audio file."""
    batch_size = 4  # Adjust if running out of memory

    audio = load_audio(audio_file)
    trans_result = whisper_model.transcribe(audio, batch_size=batch_size, language="en")

    try:
        align_result = align_for_prosody_features(
            trans_result["segments"],
            alignment_model,
            alignmet_model_metadata,
            audio,
            device,
            return_char_alignments=True,
        )
    except Exception as e:
        print(f"Error processing {audio_file}: {e}")
        return []

    return align_result["char_segments"]


def process_files(rank, world_size, all_audio_files, args):
    """Main function executed by each process."""
    setup(rank, world_size)
    
    # Load models on assigned GPU
    device = torch.device(f"cuda:{rank}")
    whisper_model = load_model("large-v2", device='cuda', device_index=rank, compute_type=args.compute_type, language='en') 
    alignment_model, alignmet_model_metadata = load_align_model(language_code="en", device=device)

    # Distribute files across GPUs
    local_files = all_audio_files[rank::world_size]

    bad_files = []
    bad_file_log = os.path.join(args.save_root, f'bad_files_{rank}.json')

    # Initialize tqdm on rank 0 (single shared progress bar)
    if rank == 0:
        pbar = tqdm.tqdm(total=len(all_audio_files), desc="Processing Progress (All GPUs)", position=0, leave=True)

    local_processed_count = 0  # Local counter for processed files

    for audio_file_path, save_path in local_files:
        if os.path.exists(save_path) and args.skip_existing:
            continue

        aligned_chars = get_aligned_chars(
            whisper_model=whisper_model,
            alignment_model=alignment_model,
            alignmet_model_metadata=alignmet_model_metadata,
            audio_file=audio_file_path,
            device=device,
        )

        if not aligned_chars:
            print(f"ERROR: failed to align file {audio_file_path}")
            bad_files.append(audio_file_path)
            with open(bad_file_log, "w") as save_file:
                json.dump(bad_files, save_file)
            continue

        char_seq = generate_char_frame_sequence(aligned_chars)

        if char_seq is None:
            print(f"ERROR: failed to generate char sequence for {audio_file_path}")
            bad_files.append(audio_file_path)
            with open(bad_file_log, "w") as save_file:
                json.dump(bad_files, save_file)
            continue

        with open(save_path, "w") as save_file:
            json.dump(char_seq, save_file)

        local_processed_count += 1  # Track local progress

        # Synchronize progress across GPUs
        global_counts = [torch.tensor(0, device=device) for _ in range(world_size)]
        dist.all_gather(global_counts, torch.tensor(local_processed_count, device=device)) 

        if rank == 0:
            total_processed = sum(t.item() for t in global_counts)  # Sum progress from all GPUs
            pbar.n = total_processed  # Update tqdm
            pbar.refresh()

    if rank == 0:
        pbar.close()

    cleanup()


if __name__ == "__main__":
    # Argument parser setup
    parser = argparse.ArgumentParser(description="Feature extraction script with alignment.")
    parser.add_argument("--data-root", type=str, help="Root data directory.")
    parser.add_argument("--save-root", type=str, help="Root directory for saving prosody features.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for model inference.")
    parser.add_argument("--compute-type", type=str, default="float32", help="Compute format type.")
    parser.add_argument("--file-type", type=str, default="wav", help="Type of audio file.")
    parser.add_argument("--skip-existing", action="store_true", help="Skip processing of existing files.")
    parser.add_argument("--chunk-file", type=str, required=False, help="JSON file containing all chunks.")
    parser.add_argument("--chunk-index", type=int, required=False, help="Index of the chunk to process.")
    args = parser.parse_args()

    # Locate audio files
    all_audio_files = []
    if args.chunk_file:
        assert args.chunk_index is not None, "chunk-index must be provided if chunk_file is specified."
        print("Extracting chunk index:", args.chunk_index)
        with open(args.chunk_file, "r") as f:
            all_chunks = json.load(f)
        chunk_files = all_chunks[args.chunk_index]
        
        for audio_file_path in chunk_files:
            rel_path = os.path.relpath(audio_file_path, args.data_root)
            save_path = os.path.join(args.save_root, rel_path.replace(args.file_type, "json"))
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            all_audio_files.append((audio_file_path, save_path))
    else:
        for dirpath, _, filenames in os.walk(args.data_root):
            rel_path = os.path.relpath(dirpath, args.data_root)
            save_dir_path = os.path.join(args.save_root, rel_path)
            os.makedirs(save_dir_path, exist_ok=True)

            for file in filenames:
                if file.endswith(args.file_type):
                    audio_file_path = os.path.join(dirpath, file)
                    save_path = os.path.join(save_dir_path, file.replace(args.file_type, "json"))
                    all_audio_files.append((audio_file_path, save_path))

    # Get world size (number of GPUs available)
    world_size = torch.cuda.device_count()
    print(f"Found {world_size} GPUs for distributed processing.")
    if world_size < 1:
        raise RuntimeError("No GPUs found for distributed processing.")
    
    # Spawn multiple processes
    mp.spawn(process_files, args=(world_size, all_audio_files, args), nprocs=world_size, join=True)
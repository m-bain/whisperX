import os
from typing import List
import whisperx
from whisperx.prosody_features.utils import generate_char_frame_sequence
import gc 
import tqdm

def get_aligned_chars(audio_file: str, device: str = 'cpu') -> List[dict]:

    batch_size = 16 # reduce if low on GPU mem
    compute_type = "float32" # change to "int8" if low on GPU mem (may reduce accuracy)

    # 1. Transcribe with original whisper (batched)
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)

    # save model to local path (optional)
    # model_dir = "/path/"
    # model = whisperx.load_model("large-v2", device, compute_type=compute_type, download_root=model_dir)

    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size)

    # delete model if low on GPU resources
    # import gc; gc.collect(); torch.cuda.empty_cache(); del model

    # 2. Align whisper output
    model_a, metadata = whisperx.load_align_model(language_code=result["language"], device=device)
    result = whisperx.align_for_prosody_features(result["segments"], model_a, metadata, audio, device, return_char_alignments=True)

    return result['char']

if __name__ == "__main__":

    root = "/project/shrikann_35/nmehlman/vpc"
    device = "cuda"

    for dirpath, dirnames, filenames in os.walk(root):
        
        if 'wav' in dirpath: 
            
            for file in tqdm.tqdm(
                os.listdir(dirpath),
                desc='extracting features'
            ): # For each audio file
                
                full_path = os.path.join(dirpath, file)

                aligned_chars = get_aligned_chars(audio_file=full_path, device=device)

                char_seq = generate_char_frame_sequence(aligned_chars)
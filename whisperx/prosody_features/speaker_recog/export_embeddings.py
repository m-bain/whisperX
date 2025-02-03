from whisperx.prosody_features.speaker_recog.sr_model import SpeakerRecogModel
from transformers import Wav2Vec2FeatureExtractor
import os
import torchaudio
import tqdm
import torch
import argparse

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Export speaker embeddings from audio files.")
    parser.add_argument("--data_root", type=str, required=True, help="Root directory of the audio data.")
    parser.add_argument("--save_root", type=str, required=True, help="Directory to save the embeddings.")
    parser.add_argument("--model_name", type=str, default="wavlm", help="Name of the model to use.")
    parser.add_argument("--file_type", type=str, default="flac", help="Type of audio files to process.")
    parser.add_argument("--skip_existing", action="store_true", help="Skip processing if embeddings already exist.")
    parser.add_argument("--device", type=str, default="cuda", help="Device to run the model on.")
    
    args = parser.parse_args()
    
    data_root = args.data_root
    model_name = args.model_name
    file_type = args.file_type
    save_root = args.save_root
    skip_existing = args.skip_existing
    device = args.device
    
    if model_name == "wavlm":
        feature_extractor = Wav2Vec2FeatureExtractor.from_pretrained('microsoft/wavlm-base-sv')
    else:
        raise ValueError("Model name not recognized")
    
    model = SpeakerRecogModel(model_name, num_speakers=2).to(device) # num_speakers is a dummy value
            
    # Collect all audio file paths
    all_audio_files = []
    for dirpath, dirnames, filenames in os.walk(data_root):
        rel_path = os.path.relpath(dirpath, data_root)
        save_dir_path = os.path.join(save_root, rel_path)

        audio_files = [f for f in filenames if f.endswith(file_type)]
        for file in audio_files:
            audio_file_path = os.path.join(dirpath, file)
            save_path = os.path.join(save_dir_path, file.replace(file_type, f"{model_name}.pt"))
            all_audio_files.append((audio_file_path, save_path))
            
    for audio_file_path, save_path in tqdm.tqdm(all_audio_files):
        
        x, _ = torchaudio.load(audio_file_path)
        x = x.squeeze().numpy()
        
        x_prc = feature_extractor(x, sampling_rate=16000, return_tensors="pt").input_values
        x_prc = x_prc.to(device)
        z = model.get_embeddings(x_prc).cpu()
    
        torch.save(z, save_path)
        

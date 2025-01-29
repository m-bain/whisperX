from whisperx.prosody_features.speaker_recog.sr_model import SpeakerRecogModel
from transformers import Wav2Vec2FeatureExtractor
import os
import torchaudio
import tqdm

if __name__ == "__main__":
    
    data_root: str = "/project/shrikann_35/nmehlman/psid_data/LibriSpeech/train-other-500"
    model_name: str = "wavlm"
    file_type: str = "flac"
    save_root: str = "/project/shrikann_35/nmehlman/psid_data/librispeech_feats/"
    device: str = "cuda"
    
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
            save_path = os.path.join(save_dir_path, file.replace(file_type, "pt"))
            all_audio_files.append((audio_file_path, save_path))
            
    for audio_file_path, save_path in tqdm.tqdm(all_audio_files):
        
        x, _ = torchaudio.load(audio_file_path)
        x = x.squeeze().numpy()
        
        x_prc = feature_extractor(x, sampling_rate=16000, return_tensors="pt").input_values
        x_prc = x_prc.to(device)
        z = model.get_embeddings(x_prc).cpu()
        
        

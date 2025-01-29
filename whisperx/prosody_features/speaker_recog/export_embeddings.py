from whisperx.prosody_features.speaker_recog.data.utils import get_dataloaders
import os

if __name__ == "__main__":
    
    data_root: str = "/project/shrikann_35/nmehlman/psid_data/LibriSpeech/train-other-500"
    model_name: str = "wavlm"
    file_type: str = ".flac"
    save_root: str = "/project/shrikann_35/nmehlman/psid_data/librispeech_feats/"
    
    # Collect all audio file paths
    all_audio_files = []
    for dirpath, dirnames, filenames in os.walk(data_root):
        rel_path = os.path.relpath(dirpath, data_root)
        save_dir_path = os.path.join(save_root, rel_path)

        audio_files = [f for f in filenames if f.endswith(file_type)]
        for file in audio_files:
            audio_file_path = os.path.join(dirpath, file)
            save_path = os.path.join(save_dir_path, file.replace(file_type, "json"))
            all_audio_files.append((audio_file_path, save_path))
            
    print(all_audio_files)
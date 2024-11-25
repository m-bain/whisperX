from torch.utils.data import Dataset, DataLoader, random_split
import torch
from typing import List, Tuple
import os
import json
from whisperx.prosody_features.tokenizer import CharLevelTokenizer

VC_SYSTEMS = ("B3", "B4", "B5", "T10-2", "T12-5", "T25-1",  "T8-5")
VALID_SPLITS = ("libri_dev_enrolls", "libri_test_enrolls", "train-clean-360", "libri_dev_trials_f", "libri_test_trials_f", "libri_dev_trials_m", "libri_test_trials_m")

class VPCDataset(Dataset):
    
    def __init__(self, root_path: str, tokenizer: CharLevelTokenizer, system: str = "all", split = "train-clean-360"):
        
        self.root_path = root_path
        self.system = system
        self.split = split
        self.tokenizer = tokenizer
        
        assert split in VALID_SPLITS, f'Invalid split. Must be one of {VALID_SPLITS}'
                
        if system == 'all': # Train on all VC systems
            
            self.paths = []
            self.speakers = []
            
            for sys in VC_SYSTEMS: # For each system
                
                paths, speakers = self._build_system_data_paths(system=sys)
                self.path += paths
                self.speakers += speakers
            
        else: # Single VC system
            self.paths, self.speakers = self._build_system_data_paths(system=system)
        
        # Renumber speakers to be sequential
        self._renumber_speakers()
            
    def _build_system_data_paths(self, system: str) -> Tuple[List[str], List[int]]:
    
        sys_data_dir = os.path.join(self.root_path, system, "data", f"{self.split}_{system}")
        utt_to_speak_path = os.path.join(sys_data_dir, "utt2spk")
        feats_dir = os.path.join(sys_data_dir, "char_feats")
        
        # Create dict to map from sample ID to speaker
        utt_to_speak = {line.split()[0]: line.split()[1] for line in open(utt_to_speak_path).readlines()}
        
        paths = []
        speakers = []
        for feat_file in os.listdir(feats_dir): # For each feature
            
            full_file_path = os.path.join(feats_dir, feat_file)
            
            id = feat_file.replace('.json', '')
            speaker = utt_to_speak[id]
            
            # Add path and data info
            paths.append(full_file_path)
            speakers.append(int(speaker))
            
        return paths, speakers
    
    def _renumber_speakers(self):
        
        unique_speakers = sorted(list(set(self.speakers)))
        speaker_id_map = {old_id: i for i, old_id in enumerate(unique_speakers)} # Assign new sequential numbering
        
        # Compute total number of unique speakers
        self.num_speakers = len(unique_speakers)
        print(f'Found {self.num_speakers} total speakers')
        
        self.speakers = [speaker_id_map[id] for id in self.speakers]
        
    def total_speakers(self) -> int:
        return self.num_speakers
    
    def __len__(self) -> int:
        return len(self.paths)
    
    def __getitem__(self, index) -> Tuple[torch.Tensor, int]:
        
        path, speaker = self.paths[index], self.speakers[index]
        
        char_seq = json.load(open(path)) # Load
        tokens = self.tokenizer.encode(char_seq) # Convert to tokens
        
        return tokens, speaker
        
        
        
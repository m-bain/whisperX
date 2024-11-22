from torch.utils.data import Dataset, DataLoader, random_split
import torch
import os

VC_SYSTEMS = ("B3", "B4", "B5", "T10-2", "T12-5", "T25-1",  "T8-5")

class VPCDataset(Dataset):
    
    def __init__(self, root_path: str, system: str = "all", split = "train-clean-360"):
        
        self.paths = []
        self.speakers = []
        
        if system == 'all': # Train on all VC systems
            pass
            
        else:
            pass
            
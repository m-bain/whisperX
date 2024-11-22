from torch.utils.data import Dataset, DataLoader, random_split
import torch
import os

class VPCDataset(Dataset):
    
    def __init__(self, root_path: str, system: str = "all", split = "train"):
        
        self.paths = []
        self.speakers = []
        
        
        
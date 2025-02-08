from torch.utils.data import Dataset, DataLoader, random_split
import torch
from typing import List, Tuple, Dict, Union
import os
import json
from whisperx.prosody_features.tokenizer import CharLevelTokenizer

class ProsodyDataset(Dataset):
    """
    Dataset for LibriSpeech with character-level features.

    Args:
        root_path (str): Path to the root directory containing data.
        tokenizer (CharLevelTokenizer): Tokenizer for encoding character sequences.
        split (str): Dataset split to use. Must be one of VALID_SPLITS.

    Raises:
        AssertionError: If the specified split is not valid.
    """

    def __init__(
        self,
        root_path: str,
        tokenizer: CharLevelTokenizer,
        split: str = "train",
        max_sample_length: int = 1024,
        sr_embed_model: str | None = None,
    ):
        self.root_path = root_path
        self.split = split
        self.tokenizer = tokenizer
        self.max_sample_length = max_sample_length
        self.sr_embed_model = sr_embed_model

        splits_path = os.path.join(root_path, "splits.json")
        splits = json.load(open(splits_path))
        assert self.split in splits, f"Split {self.split} not found in splits.json"

        # Load data paths and speaker labels
        self.samples = splits[self.split]

        # Filter out samples that don't have a corresponding .json file
        _samples = [sample for sample in self.samples if os.path.exists(sample["path"])]

        if len(_samples) != len(self.samples):
            print(
                f"Filtered out {len(self.samples) - len(_samples)} samples without .json files"
            )   
        
        self.samples = _samples

        # Renumber speakers to ensure they are sequential
        self._renumber_speakers()

    def _renumber_speakers(self):
        """
        Renumber speakers to ensure IDs are sequential and compute the total number of unique speakers.
        """
        unique_speakers = sorted(
            list(set([sample["speaker"] for sample in self.samples]))
        )
        self.speaker_id_map = {
            old_id: i for i, old_id in enumerate(unique_speakers)
        }  # Map old IDs to new ones

        self.num_speakers = len(unique_speakers)  # Total number of unique speakers

        print(f"Found {self.num_speakers} total speakers")

    def total_speakers(self) -> int:
        """
        Get the total number of unique speakers.

        Returns:
            int: Total unique speakers in the dataset.
        """
        return self.num_speakers

    def __len__(self) -> int:
        """
        Get the total number of data samples.

        Returns:
            int: Total number of data samples.
        """
        return len(self.samples)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Get a data sample and its corresponding speaker ID.

        Args:
            index (int): Index of the data sample.

        Returns:
            Tuple[torch.Tensor, int]: Tokenized character sequence and speaker ID.
        """
        sample = self.samples[index]

        path = sample["path"]
        speaker_raw = sample["speaker"]
        speaker_id = self.speaker_id_map[speaker_raw]

        # Load character sequence and tokenize
        char_seq = json.load(open(path))
        tokens = self.tokenizer.encode(char_seq)
        
        if len(tokens) > self.max_sample_length:
            tokens = tokens[:self.max_sample_length]
        
        if self.sr_embed_model:
            embed_path = path.replace('.json', f'.{self.sr_embed_model}.pt')
            sr_embeds = torch.load(embed_path, weights_only=False)
            return tokens, sr_embeds.squeeze(), speaker_id
        else:
            return tokens, speaker_id


if __name__ == "__main__":

    import numpy as np

    tokenizer = CharLevelTokenizer()
    dataset = ProsodyDataset(
        root_path="/project/shrikann_35/nmehlman/psid_data/vox1_feats",
        tokenizer=tokenizer,
        sr_embed_model='wavlm'
    )

    idx = np.random.randint(len(dataset))
    tokens, embeds, speaker_id = dataset[idx]
    print(f"Sample {idx} - Speaker ID: {speaker_id}")
    print(f"Tokens: {tokens}")
    print(f"Embeds shape: {embeds.shape}")
    print(f"Tokens shape: {tokens.shape}")

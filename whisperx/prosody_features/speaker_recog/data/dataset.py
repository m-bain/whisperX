from torch.utils.data import Dataset, DataLoader, random_split
import torch
import torchaudio
from typing import Tuple
import os
import json

class SpeakerRecogDataset(Dataset):

    def __init__(
        self,
        root_path: str,
        split: str = "train",
    ):
        self.root_path = root_path
        self.split = split

        splits_path = os.path.join(self.root_path, "splits.json")
        splits = json.load(open(splits_path))
        assert self.split in splits, f"Split {self.split} not found in splits.json"

        # Load data paths and speaker labels
        self.samples = splits[self.split]

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
        audio = torchaudio.load(path)[0].squeeze().numpy()

        return audio, speaker_id


if __name__ == "__main__":

    import numpy as np

    dataset = SpeakerRecogDataset(
        root_path="/project/shrikann_35/nmehlman/psid_data/LibriSpeech/train-other-500",
    )

    idx = np.random.randint(len(dataset))
    audio, speaker_id = dataset[idx]
    print(f"Sample {idx} - Speaker ID: {speaker_id}")
    print(f"Audio: {audio}")
    print(f"Audio shape: {audio.shape}")

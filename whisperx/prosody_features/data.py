from torch.utils.data import Dataset, DataLoader, random_split
import torch
from typing import List, Tuple, Dict, Union
import os
import json
from whisperx.prosody_features.tokenizer import CharLevelTokenizer

VC_SYSTEMS = ("B3", "B4", "B5", "T10-2", "T12-5", "T25-1", "T8-5")
VALID_SPLITS = (
    "libri_dev_enrolls",
    "libri_test_enrolls",
    "train-clean-360",
    "libri_dev_trials_f",
    "libri_test_trials_f",
    "libri_dev_trials_m",
    "libri_test_trials_m",
)


class VPCDataset(Dataset):
    """
    Dataset for Voice Conversion (VC) systems with character-level features.

    Args:
        root_path (str): Path to the root directory containing data.
        tokenizer (CharLevelTokenizer): Tokenizer for encoding character sequences.
        system (str): Specific VC system to use or "all" for all systems. Defaults to "all".
        split (str): Dataset split to use. Must be one of VALID_SPLITS.

    Raises:
        AssertionError: If the specified split is not valid.
    """

    def __init__(
        self,
        root_path: str,
        tokenizer: CharLevelTokenizer,
        system: str = "all",
        split: str = "train-clean-360",
        return_id: bool = False
    ):
        self.root_path = root_path
        self.system = system
        self.split = split
        self.tokenizer = tokenizer
        self.return_id = return_id

        # Validate the split
        assert split in VALID_SPLITS, f"Invalid split. Must be one of {VALID_SPLITS}"

        # Handle system selection
        if system == "all":  # Train on all VC systems
            self.paths = []
            self.speakers = []
            for sys in VC_SYSTEMS:  # Process each system
                paths, speakers = self._build_system_data_paths(system=sys)
                self.paths += paths
                self.speakers += speakers
        else:  # Single VC system
            self.paths, self.speakers = self._build_system_data_paths(system=system)

        # Renumber speakers to ensure they are sequential
        self._renumber_speakers()

    def _build_system_data_paths(self, system: str) -> Tuple[List[str], List[int]]:
        """
        Build data paths and speaker labels for a specific system.

        Args:
            system (str): VC system identifier.

        Returns:
            Tuple[List[str], List[int]]: File paths and corresponding speaker labels.
        """
        sys_data_dir = os.path.join(
            self.root_path, system, "data", f"{self.split}_{system}"
        )
        utt_to_speak_path = os.path.join(sys_data_dir, "utt2spk")
        feats_dir = os.path.join(sys_data_dir, "char_feats")

        # Map utterance IDs to speaker IDs
        utt_to_speak = {
            line.split()[0]: int(line.split()[1])
            for line in open(utt_to_speak_path).readlines()
        }

        paths, speakers = [], []
        for feat_file in os.listdir(feats_dir):  # For each feature file
            full_file_path = os.path.join(feats_dir, feat_file)
            utt_id = feat_file.replace(".json", "")
            speaker = utt_to_speak[utt_id]
            paths.append(full_file_path)
            speakers.append(speaker)

        return paths, speakers

    def _renumber_speakers(self):
        """
        Renumber speakers to ensure IDs are sequential and compute the total number of unique speakers.
        """
        unique_speakers = sorted(list(set(self.speakers)))
        speaker_id_map = {
            old_id: i for i, old_id in enumerate(unique_speakers)
        }  # Map old IDs to new ones

        # Update speaker IDs to be sequential
        self.speakers = [speaker_id_map[speaker] for speaker in self.speakers]
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
        return len(self.paths)

    def __getitem__(self, index: int) -> Tuple[torch.Tensor, int]:
        """
        Get a data sample and its corresponding speaker ID.

        Args:
            index (int): Index of the data sample.

        Returns:
            Tuple[torch.Tensor, int]: Tokenized character sequence and speaker ID.
        """
        path, speaker = self.paths[index], self.speakers[index]

        id = path.split('/')[-1].replace('json', '')

        # Load character sequence and tokenize
        char_seq = json.load(open(path))
        tokens = self.tokenizer.encode(char_seq)

        if len(tokens) > 5000:
            print('WARNING: truncating token sequence (exceeds max length)')
            tokens = tokens[:5000]

        if self.return_id:
            return tokens, id
        else:
            return tokens, speaker


def collate_fn(
    batch: List[Tuple[torch.Tensor, int]]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Collate function to pad sequences to the same length for batching.

    Args:
        batch (List[Tuple[torch.Tensor, int]]): A batch of data samples, where each sample is a tuple of
                                                (sequence tensor, speaker ID).

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: A tuple containing:
            - Padded sequences (torch.Tensor) of shape (batch_size, max_seq_len).
            - Speaker IDs (torch.Tensor) of shape (batch_size).
    """
    # Separate sequences and speaker IDs
    sequences, speaker_ids = zip(*batch)

    # Find the length of the longest sequence in the batch
    max_seq_len = max(seq.size(0) for seq in sequences)

    # Initialize a tensor for padded sequences with zeros
    padded_sequences = torch.zeros(len(sequences), max_seq_len, dtype=torch.long)

    # Copy each sequence into the padded tensor
    for i, seq in enumerate(sequences):
        padded_sequences[i, : seq.size(0)] = seq  # Copy the sequence up to its length

    # Convert speaker IDs to a tensor
    speaker_ids = torch.tensor(speaker_ids, dtype=torch.long)

    return padded_sequences, speaker_ids


def get_dataloaders(
    root_path: str,
    tokenizer: CharLevelTokenizer,
    system: str,
    split: str,
    train_frac: float = 1.0,
    batch_size: int = 16,
    num_workers: int = 1,
    shuffle: bool = True,
    **dataloader_kwargs,
) -> Union[DataLoader, Dict[str, DataLoader]]:
    """
    Create DataLoaders for training and validation.

    Args:
        root_path (str): Path to the dataset root.
        tokenizer (CharLevelTokenizer): Tokenizer for encoding character sequences.
        system (str): VC system to use or "all".
        split (str): Dataset split to use.
        train_frac (float): Fraction of data for training. Defaults to 1.0.
        batch_size (int): Batch size for DataLoader. Defaults to 16.
        num_workers (int): Number of workers for DataLoader. Defaults to 1.
        shuffle (bool): Whether to shuffle the data. Defaults to True.
        **dataloader_kwargs: Additional arguments for DataLoader.

    Returns:
        Union[DataLoader, Dict[str, DataLoader]]: A dict with "train" and (possibly) "val" DataLoaders.
    """
    full_dataset = VPCDataset(
        root_path=root_path, tokenizer=tokenizer, system=system, split=split
    )
    total_speakers = full_dataset.total_speakers()

    if train_frac < 1.0:  # Create a validation split
        train_size = int(train_frac * len(full_dataset))
        val_size = len(full_dataset) - train_size

        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Build dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=32,
            num_workers=num_workers,
            shuffle=False,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )

        # Store number of speakers for easy access
        train_dataloader.total_speakers = total_speakers
        val_dataloader.total_speakers = total_speakers

        return {"train": train_dataloader, "val": val_dataloader}
    else:  # Train on the full dataset
        train_dataloader = DataLoader(
            full_dataset,
            batch_size=batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )
        train_dataloader.total_speakers = total_speakers
        return {"train": train_dataloader}

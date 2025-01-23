import torch
from torch.utils.data import DataLoader, random_split
from typing import List, Literal, Tuple, Union, Dict

from whisperx.prosody_features.tokenizer import CharLevelTokenizer
from whisperx.prosody_features.data.dataset import ProsodyDataset


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
    if isinstance(speaker_ids[0], str):
        speaker_ids = [id for id in speaker_ids]
    else:
        speaker_ids = torch.tensor(speaker_ids, dtype=torch.long)

    return padded_sequences, speaker_ids


def get_dataloaders(
    root_path: str,
    tokenizer: CharLevelTokenizer,
    split: str,
    val_frac: float = 0.0,
    train_batch_size: int = 16,
    val_batch_size: int = 32,
    num_workers: int = 1,
    shuffle: bool = True,
    **dataloader_kwargs,
) -> Union[DataLoader, Dict[str, DataLoader]]:
    """
    Create DataLoaders for training and validation.

    Args:
        root_path (str): Path to the dataset root.
        tokenizer (CharLevelTokenizer): Tokenizer for encoding character sequences.
        split (str): Dataset split to use.
        val_frac (float, optional): Fraction of data for validation. Defaults to 0.0.
        train_batch_size (int, optional): Batch size for training DataLoader. Defaults to 16.
        val_batch_size (int, optional): Batch size for validation DataLoader. Defaults to 32.
        num_workers (int, optional): Number of workers for DataLoader. Defaults to 1.
        shuffle (bool, optional): Whether to shuffle the training data. Defaults to True.
        **dataloader_kwargs: Additional arguments for DataLoader.

    Returns:
        Union[DataLoader, Dict[str, DataLoader]]: A dictionary with "train" and (optionally) "val" DataLoaders.

    Raises:
        ValueError: If the specified dataset is not supported.
    """

    full_dataset = ProsodyDataset(
        root_path=root_path, tokenizer=tokenizer, split=split
    )

    total_speakers = full_dataset.total_speakers()

    if val_frac > 0.0:  # Create a validation split if requested
        val_size = int(val_frac * len(full_dataset))
        train_size = len(full_dataset) - val_size

        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        # Build dataloaders
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=train_batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=val_batch_size,
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
            batch_size=train_batch_size,
            num_workers=num_workers,
            shuffle=shuffle,
            collate_fn=collate_fn,
            **dataloader_kwargs,
        )
        train_dataloader.total_speakers = total_speakers
        return {"train": train_dataloader}

if __name__ == "__main__":

    # Define parameters for testing
    dataset = "librispeech"
    root_path = "/project/shrikann_35/nmehlman/psid_data/librispeech"
    split = "train"
    val_frac = 0.1
    train_batch_size = 16
    val_batch_size = 32
    num_workers = 1
    shuffle = True

    # Initialize tokenizer
    tokenizer = CharLevelTokenizer()

    # Get dataloaders
    dataloaders = get_dataloaders(
        dataset=dataset,
        root_path=root_path,
        tokenizer=tokenizer,
        split=split,
        val_frac=val_frac,
        train_batch_size=train_batch_size,
        val_batch_size=val_batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
    )

    # Print information about the dataloaders
    print("Train DataLoader:", dataloaders["train"])
    if "val" in dataloaders:
        print("Validation DataLoader:", dataloaders["val"])

    # Load a test batch from each DataLoader and print data and label shapes
    train_batch = next(iter(dataloaders["train"]))
    print("Train batch data shape:", train_batch[0].shape)
    print("Train batch label shape:", train_batch[1].shape)

    if "val" in dataloaders:
        val_batch = next(iter(dataloaders["val"]))
        print("Validation batch data shape:", val_batch[0].shape)
        print("Validation batch label shape:", val_batch[1].shape)
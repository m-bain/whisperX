import os
import torch
from torch.utils.data import DataLoader
from whisperx.prosody_features.feature_model import ProsodySpeakerVerificationModel
from whisperx.prosody_features.utils import load_yaml_config
from whisperx.prosody_features.data import get_dataloaders
from whisperx.prosody_features.tokenizer import CharLevelTokenizer
import tqdm


def extract_and_save_embeddings(
    model_checkpoint: str,
    dataloader: DataLoader,
    output_dir: str,
    device: str = "cuda"
):
    """
    Extracts embeddings using a pre-trained model and saves them to a specified directory.

    Args:
        model_checkpoint (str): Path to the model checkpoint file.
        dataloader (DataLoader): DataLoader providing audio data and speaker labels.
        output_dir (str): Directory to save the embeddings and labels.
        device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
        None: Saves the embeddings and labels to the output directory.
    """
    # Ensure the output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Load the model
    print("Loading model...")
    model = ProsodySpeakerVerificationModel.load_from_checkpoint(
        model_checkpoint, map_location=device
    )
    model.eval().to(device)

    # Extract embeddings
    print("Extracting embeddings...")
    for audio, ids in tqdm.tqdm(dataloader, desc="Processing batches"):
        
        audio = audio.to(device)
        embeddings = model.get_features(audio).cpu()

        for embed, id in zip(embeddings, ids):
            torch.save(embed, os.path.join(output_dir, f"{id}.pt"))

    print(f"Embeddings successfully saved to {output_dir}")

if __name__ == "__main__":
    import sys

    device = "cpu"

    # Load configuration
    config_path = sys.argv[1]

    print("Loading configuration...")
    config = load_yaml_config(config_path)

    # Extract system-specific configuration
    system = config["dataset"]["system"]
    assert system != "all", "System cannot be 'all'. Please specify a valid system."

    # Prepare output directory
    output_dir = os.path.join(config["output_dir"], system)

    tokenizer = CharLevelTokenizer()

    # Get dataloaders
    dataloaders = get_dataloaders(
        tokenizer=tokenizer, return_id=True, **config["dataset"], **config["dataloader"]
    )
    dataloader = dataloaders["train"]  # Assuming "train" contains the relevant data

    # Extract and save embeddings
    extract_and_save_embeddings(
        model_checkpoint=config["ckpt_path"],
        dataloader=dataloader,
        output_dir=output_dir,
        device=device,
    )

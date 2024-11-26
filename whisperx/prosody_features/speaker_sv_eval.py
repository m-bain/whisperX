from whisperx.prosody_features.feature_model import ProsodySpeakerVerificationModel
from torch.utils.data import DataLoader
import os
import torch
import tqdm


def extract_enrollment_embeddings(
    ckpt_path: str,
    embedding_dataloader: DataLoader,
    embedding_dir: str,
    device: str = "cpu",
):

    model = ProsodySpeakerVerificationModel.load_from_checkpoint(
        ckpt_path, map_location=device
    )

    for i, (audio, spk) in tqdm.tqmd(
        enumerate(embedding_dataloader), desc="extracting enrollment embeddings"
    ):  # For each sample

        z = model.get_features(audio)

        spk_dir = os.path.join(embedding_dir, f"speaker_{spk}")
        if not os.path.exists(spk_dir):  # Make directory if needed
            os.mkdir(spk_dir)

        # Save
        save_path = os.path.join(spk_dir, f"sample_{i}.pt")
        torch.save(z, save_path)

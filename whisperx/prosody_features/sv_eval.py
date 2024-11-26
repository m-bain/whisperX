from whisperx.prosody_features.feature_model import ProsodySpeakerVerificationModel
from torch.utils.data import DataLoader
import os
import torch
import tqdm
from typing import Tuple
from speechbrain.processing.PLDA_LDA import *


def extract_enrollment_embeddings(
    model: ProsodySpeakerVerificationModel,
    dataloader: DataLoader,
    device: str = "cpu",
):
    """
    Extracts speaker embeddings from audio data.

    Args:
        model (ProsodySpeakerVerificationModel): Pretrained model.
        dataloader (DataLoader): DataLoader providing audio data and speaker IDs.
        device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
        None
    """

    embeds, labels = [], []

    for i, (audio, spk) in tqdm.tqmd(
        enumerate(dataloader), desc="extracting enrollment embeddings"
    ):  # For each sample

        audio = audio.to(device)
        z = model.get_features(audio).cpu()

        embeds.append(z)
        labels.append(spk)

    embeds = torch.concat(embeds, 0)
    labels = torch.concat(labels, 0)

    return embeds, labels


def maybe_load_or_generate_embeds(
    model: ProsodySpeakerVerificationModel,
    dataloader: DataLoader | None = None,
    embed_dir: str | None = None,
    should_save: bool = False,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:

    should_generate_embeds = embed_dir is None

    if embed_dir:  # Try to load pre-saved embeddings
        print(f"Attempting to load saved embeddings from {embed_dir}")
        try:
            embeds = torch.load(os.path.join(embed_dir, "embeds.pt"))
            labels = torch.load(os.path.join(embed_dir, "labels.pt"))
        except FileNotFoundError:  # Embeddings not found
            print(f"Saved embeddings NOT found in {embed_dir}")
            should_generate_embeds = True

    if should_generate_embeds:  # Get embeddings from model
        assert dataloader is not None, "no dataloader provided"
        embeds, labels = extract_enrollment_embeddings(
            model=model, dataloader=dataloader, device=device
        )

    if should_save:
        assert embed_dir is not None, "embed_dir is required to save embeddings"
        torch.save(embeds, os.path.join(embed_dir, "embeds.pt"))
        torch.save(labels, os.path.join(embed_dir, "labels.pt"))

    return embeds, labels


def fit_and_score_plda(
    enroll_embeds: torch.Tensor,
    enroll_labels: torch.Tensor,
    test_embeds: torch.Tensor,
    test_labels: torch.Tensor,
):

    pass


def run_speaker_verification_eval(
    ckpt_path: str,
    enroll_dataloader: DataLoader | None = None,
    test_dataloader: DataLoader | None = None,
    device: str = "cpu",
    enroll_embed_dir: str | None = None,
    test_embed_dir: str | None = None,
    save_embeds: bool = False,
    **plda_kwargs,
):

    # Load pre-trained model from checkpoint
    model = ProsodySpeakerVerificationModel.load_from_checkpoint(
        ckpt_path, map_location=device
    )

    # Get enrollment embeddings
    enroll_embeds, enroll_labels = maybe_load_or_generate_embeds(
        model=model,
        dataloader=enroll_dataloader,
        embed_dir=enroll_embed_dir,
        should_save=save_embeds,
        device=device,
    )

    # Get test embeddings
    test_embeds, test_labels = maybe_load_or_generate_embeds(
        model=model,
        dataloader=test_dataloader,
        embed_dir=test_embed_dir,
        should_save=save_embeds,
        device=device,
    )

    plda = PLDA()

    # Train the PLDA model
    plda.train_plda(embeddings, speaker_ids)

    # Perform scoring with PLDA
    scores = plda.score_plda(enrollment_embeddings, test_embeddings)

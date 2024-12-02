from whisperx.prosody_features.feature_model import ProsodySpeakerVerificationModel
from whisperx.prosody_features.utils import average_2d_by_labels
from torch.utils.data import DataLoader
import os
import torch
import tqdm
from typing import Tuple, Union

# Coppied from https://speechbrain.readthedocs.io/en/latest/_modules/speechbrain/utils/metric_stats.html#EER
def EER(positive_scores: torch.Tensor, negative_scores: torch.Tensor) -> Tuple[float, float]:
    """
    Computes the Equal Error Rate (EER) and the corresponding threshold.

    Args:
        positive_scores (torch.Tensor): Scores for positive samples.
        negative_scores (torch.Tensor): Scores for negative samples.

    Returns:
        Tuple[float, float]: The EER and the threshold.
    """
    thresholds, _ = torch.sort(torch.cat([positive_scores, negative_scores]))
    thresholds = torch.unique(thresholds)

    # Adding intermediate thresholds
    intermediate_thresholds = (thresholds[:-1] + thresholds[1:]) / 2
    thresholds, _ = torch.sort(torch.cat([thresholds, intermediate_thresholds]))

    min_index, final_FRR, final_FAR = 0, 0, 0

    for i, cur_thresh in enumerate(thresholds):
        FRR = (positive_scores <= cur_thresh).sum().float() / positive_scores.shape[0]
        FAR = (negative_scores > cur_thresh).sum().float() / negative_scores.shape[0]

        if abs(FAR - FRR) < abs(final_FAR - final_FRR) or i == 0:
            min_index, final_FRR, final_FAR = i, FRR.item(), FAR.item()

    EER = (final_FAR + final_FRR) / 2
    return float(EER), float(thresholds[min_index])

def extract_embeddings(
    model: ProsodySpeakerVerificationModel,
    dataloader: DataLoader,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Extracts speaker embeddings from audio data.

    Args:
        model (ProsodySpeakerVerificationModel): Pretrained model.
        dataloader (DataLoader): DataLoader providing audio data and speaker IDs.
        device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Speaker embeddings and labels.
    """
    embeds, labels = [], []

    for audio, spk in tqdm.tqdm(dataloader, desc="Extracting embeddings"): # For each batch
        audio = audio.to(device)
        z = model.get_features(audio).cpu()

        embeds.append(z)
        labels.append(spk)

    embeds = torch.cat(embeds, 0)
    labels = torch.cat(labels, 0)
    return embeds, labels

def maybe_load_or_generate_embeds(
    model: ProsodySpeakerVerificationModel,
    dataloader: Union[DataLoader, None] = None,
    embed_dir: Union[str, None] = None,
    should_save: bool = False,
    device: str = "cpu",
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Loads or generates embeddings based on availability.

    Args:
        model (ProsodySpeakerVerificationModel): Pretrained model.
        dataloader (Union[DataLoader, None]): DataLoader for generating embeddings.
        embed_dir (Union[str, None]): Directory containing saved embeddings.
        should_save (bool): Whether to save generated embeddings.
        device (str): Device to run the model on ('cpu' or 'cuda').

    Returns:
        Tuple[torch.Tensor, torch.Tensor]: Embeddings and labels.
    """
    should_generate_embeds = embed_dir is None

    if embed_dir:
        print(f"Attempting to load saved embeddings from {embed_dir}")
        try: # Try to locate saved embeddings
            embeds = torch.load(os.path.join(embed_dir, "embeds.pt"), weights_only=False)
            labels = torch.load(os.path.join(embed_dir, "labels.pt"), weights_only=False)
        except FileNotFoundError: # Not found, need to generate
            print(f"Saved embeddings NOT found in {embed_dir}")
            should_generate_embeds = True

    if should_generate_embeds:
        assert dataloader is not None, "Dataloader must be provided to generate embeddings."
        embeds, labels = extract_embeddings(model, dataloader, device)

    if should_save: # Save embeddings for future loading
        assert embed_dir is not None, "Embed directory must be specified to save embeddings."
        torch.save(embeds, os.path.join(embed_dir, "embeds.pt"))
        torch.save(labels, os.path.join(embed_dir, "labels.pt"))

    return embeds, labels

def cosine_speaker_verification(
    enroll_embeds: torch.Tensor,
    enroll_labels: torch.Tensor,
    test_embeds: torch.Tensor,
    test_labels: torch.Tensor,
) -> float:
    """
    Evaluates speaker verification using cosine similarity.

    Args:
        enroll_embeds (torch.Tensor): Enrollment embeddings.
        enroll_labels (torch.Tensor): Enrollment labels.
        test_embeds (torch.Tensor): Test embeddings.
        test_labels (torch.Tensor): Test labels.

    Returns:
        float: Equal Error Rate (EER).
    """
    mean_enroll_embeds = average_2d_by_labels(enroll_embeds, enroll_labels, axis=0) # Find average embedding for each speaker
    
    # Normalize
    mean_enroll_embeds /= torch.norm(mean_enroll_embeds, dim=1, keepdim=True)
    test_embeds /= torch.norm(test_embeds, dim=1, keepdim=True)

    # Compute cosine similiaty
    sim_mtx = test_embeds @ mean_enroll_embeds.T

    pos_mask = torch.zeros_like(sim_mtx)
    pos_mask[torch.arange(len(sim_mtx)), test_labels] = 1

    pos_scores = sim_mtx[pos_mask > 0]
    neg_scores = sim_mtx[pos_mask == 0]

    eer, _ = EER(pos_scores, neg_scores)
    
    return eer

def run_speaker_verification_eval(
    ckpt_path: str,
    num_speakers: int, 
    enroll_dataloader: Union[DataLoader, None] = None,
    test_dataloader: Union[DataLoader, None] = None,
    device: str = "cpu",
    enroll_embed_dir: Union[str, None] = None,
    test_embed_dir: Union[str, None] = None,
    save_embeds: bool = False
) -> float:
    """
    Runs the full speaker verification evaluation.

    Args:
        ckpt_path (str): Path to the model checkpoint.
        num_speakers (int): Number of speakers
        enroll_dataloader (Union[DataLoader, None]): Dataloader for enrollment data.
        test_dataloader (Union[DataLoader, None]): Dataloader for test data.
        device (str): Device to run the model on ('cpu' or 'cuda').
        enroll_embed_dir (Union[str, None]): Directory for enrollment embeddings.
        test_embed_dir (Union[str, None]): Directory for test embeddings.
        save_embeds (bool): Whether to save embeddings.
        **plda_kwargs: Additional arguments for PLDA evaluation.

    Returns:
        float: Equal Error Rate (EER).
    """
    print('Loading Model')
    model = ProsodySpeakerVerificationModel.load_from_checkpoint(
        ckpt_path, map_location=device, num_speakers=num_speakers
    )

    print('Loading or generating embeddings')
    enroll_embeds, enroll_labels = maybe_load_or_generate_embeds(
        model, enroll_dataloader, enroll_embed_dir, save_embeds, device
    )
    test_embeds, test_labels = maybe_load_or_generate_embeds(
        model, test_dataloader, test_embed_dir, save_embeds, device
    )

    print('Running EER computation')
    eer = cosine_speaker_verification(
        enroll_embeds, enroll_labels, test_embeds, test_labels
    )
    return eer

if __name__ == "__main__":

    from whisperx.prosody_features.data import get_dataloaders
    from whisperx.prosody_features.utils import load_yaml_config
    from whisperx.prosody_features.tokenizer import CharLevelTokenizer
    import sys
    
    config_path = sys.argv[1]

    config = load_yaml_config(config_path)

    tokenizer = CharLevelTokenizer()
    enroll_dataloader = get_dataloaders(tokenizer=tokenizer, **config['enroll_dataset'], **config['dataloader'])["train"]
    test_dataloader = get_dataloaders(tokenizer=tokenizer, **config['test_dataset'], **config['dataloader'])["train"]

    eer = run_speaker_verification_eval(
        enroll_dataloader=enroll_dataloader,
        test_dataloader=test_dataloader,
        **config['eval']
    )

    print(f"EER: {eer:.4f}")
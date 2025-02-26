from whisperx.prosody_features.feature_model import ProsodySpeakerIDModel
from whisperx.prosody_features.tokenizer import CharLevelTokenizer
from whisperx.prosody_features.data.utils import get_dataloaders
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import sys
from whisperx.prosody_features.utils import load_yaml_config
from pytorch_lightning.strategies import DDPStrategy
import argparse


torch.set_warn_always(False)

def main(config):
    
    # Setup dataloaders
    tokenizer = CharLevelTokenizer()
    assert tokenizer.vocab_size() == 28  # Sanity check
    dataloaders = get_dataloaders(
        tokenizer=tokenizer, **config["data"], **config["dataloader"]
    )

    num_speakers = dataloaders["train"].total_speakers
    config["lightning"]["hparams"]["max_sample_length"] = config['data']['max_sample_length']
    
    # Create Lightning module
    pl_model = ProsodySpeakerIDModel(
        num_speakers=num_speakers, **config["lightning"]
    )

    # Create logger (logs are saved to /save_dir/name/version/):
    logger = TensorBoardLogger(**config["tensorboard"])

    checkpoint_callback = ModelCheckpoint(
        monitor="val_balanced_accuracy",
        dirpath=f"{logger.log_dir}/checkpoints",  # Save in the logger's directory
        filename="best_model-{epoch:02d}-{step:04d}-{val_accuracy:.2f}",  # Include epoch, step, and val_accuracy in the filename
        save_top_k=1,
        save_last=True,
        mode="max"
    )

    ddp_strategy = DDPStrategy(find_unused_parameters=False)

    # Make trainer
    trainer = Trainer(logger=logger, callbacks=[checkpoint_callback], **config["trainer"], strategy=ddp_strategy)

    trainer.fit(
        pl_model,
        train_dataloaders=dataloaders["train"],
        val_dataloaders=dataloaders.get("val", None),
        ckpt_path=config["ckpt_path"],
    )
     

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Train a prosody ID model.")
    parser.add_argument("config_path", type=str, help="Path to the configuration file.")
    args = parser.parse_args()

    # Load config, and perform general setup
    config = load_yaml_config(args.config_path)
    
    #os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]
    if config["random_seed"]:
        pl.seed_everything(config["random_seed"], workers=True)

    main(config)
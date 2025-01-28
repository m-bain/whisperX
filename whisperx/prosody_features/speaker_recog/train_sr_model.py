from whisperx.prosody_features.speaker_recog.data.utils import get_dataloaders
from whisperx.prosody_features.speaker_recog.sr_model import SpeakerRecogModel
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import os
import sys
from whisperx.prosody_features.utils import load_yaml_config
from pytorch_lightning.strategies import DDPStrategy


torch.set_warn_always(False)

CONFIG_PATH = sys.argv[1]

def main(config):
    
    # Setup dataloaders
    dataloaders = get_dataloaders(
        **config["data"], **config["dataloader"]
    )

    num_speakers = dataloaders["train"].total_speakers
    
    # Create Lightning module
    pl_model = SpeakerRecogModel(
        num_speakers=num_speakers, **config["lightning"]
    )

    # Create logger (logs are saved to /save_dir/name/version/):
    logger = TensorBoardLogger(**config["tensorboard"])

    checkpoint_callback = ModelCheckpoint(
        monitor="val_accuracy",                          
        dirpath=f"{logger.log_dir}/checkpoints",     # Save in the logger's directory
        filename="best_model",                       
        save_top_k=1,                               
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

    # Load config, and perform general setup
    config = load_yaml_config(CONFIG_PATH)
    
    #os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]
    if config["random_seed"]:
        pl.seed_everything(config["random_seed"], workers=True)

    main(config)

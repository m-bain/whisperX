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


torch.set_warn_always(False)

CONFIG_PATH = sys.argv[1]

def main(config):
    
    # Setup dataloaders
    tokenizer = CharLevelTokenizer()
    assert tokenizer.vocab_size() == 28  # Sanity check
    dataloaders = get_dataloaders(
        tokenizer=tokenizer, **config["data"], **config["dataloader"]
    )

    config["lightning"]["hparams"]["max_sample_length"] = config['data']['max_sample_length']
    
    # Create Lightning module
    assert "checkpoint" in config, "Checkpoint path must be provided in the config"
    pl_model = ProsodySpeakerIDModel.load_from_checkpoint(config["checkpoint"])
    
    # Create logger (logs are saved to /save_dir/name/version/):
    logger = TensorBoardLogger(**config["tensorboard"])

    ddp_strategy = DDPStrategy(find_unused_parameters=False)

    # Make trainer
    trainer = Trainer(logger=logger, **config["trainer"], strategy=ddp_strategy)

    metrics = trainer.test(
        pl_model,
        dataloaders=dataloaders["test"]    
    )
    
    print(metrics)

if __name__ == "__main__":

    # Load config, and perform general setup
    config = load_yaml_config(CONFIG_PATH)
    
    #os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]
    if config["random_seed"]:
        pl.seed_everything(config["random_seed"], workers=True)

    main(config)
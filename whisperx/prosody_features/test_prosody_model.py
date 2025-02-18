from whisperx.prosody_features.feature_model import ProsodySpeakerIDModel
from whisperx.prosody_features.tokenizer import CharLevelTokenizer
from whisperx.prosody_features.data.utils import get_dataloaders
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
from pytorch_lightning.loggers import TensorBoardLogger
import os
import json
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
        tokenizer=tokenizer, **config["data"], **config["dataloader"], val_frac=0.1 # DEBUG
    )
    
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
        dataloaders=dataloaders["val"]# DEBUG dataloaders["test"]    
    )
    
    # Save metrics and config to JSON
    json_path = os.path.join(logger.log_dir, "metrics.json")
    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=4)
        
    config_save_path = os.path.join(logger.log_dir, "config.json")
    with open(config_save_path, "w") as f:
        json.dump(config, f, indent=4)
    
if __name__ == "__main__":

    # Load config, and perform general setup
    config = load_yaml_config(CONFIG_PATH)
    
    #os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]
    if config["random_seed"]:
        pl.seed_everything(config["random_seed"], workers=True)

    main(config)
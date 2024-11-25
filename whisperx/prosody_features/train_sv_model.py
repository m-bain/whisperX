from whisperx.prosody_features.feature_model import ProsodySpeakerVerificationModel
from nick_utils.nick_io import load_yaml_config
from whisperx.prosody_features.tokenizer import CharLevelTokenizer
from whisperx.prosody_features.data import get_dataloaders
import pytorch_lightning as pl
from pytorch_lightning import Trainer
import torch
from pytorch_lightning.loggers import TensorBoardLogger
import os

torch.set_warn_always(False)

CONFIG_PATH = ""

if __name__ == "__main__":

    # Load config, and perform general setup
    config = load_yaml_config(CONFIG_PATH)
    os.environ["CUDA_VISIBLE_DEVICES"] = config["gpus"]
    if config["random_seed"]:
        pl.seed_everything(config["random_seed"], workers=True)

    # Setup dataloaders
    tokenizer = CharLevelTokenizer()
    dataloaders = get_dataloaders(tokenizer=tokenizer, **config['dataset'], **config['dataloader'])

    # Create Lightning module
    #pl_model = ProsodySpeakerVerificationModel(**config["lightning"])

    # Create logger (logs are saved to /save_dir/name/version/):
    #logger = TensorBoardLogger(**config["tensorboard"])

    # Make trainer
    #trainer = Trainer(logger=logger, **config["trainer"])

    #trainer.fit(
    #    pl_model,
    #    train_dataloaders=dataloaders["train"],
    #    val_dataloaders=dataloaders["val"],
    #    ckpt_path=config["ckpt_path"],
    #)
    
    for tokens, spk in dataloaders['val']:
        print(tokens, spk)
